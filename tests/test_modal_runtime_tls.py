"""CA augmentation keeps Modal's native proxy connectors and TLS verification."""

import ssl
import sys
from types import ModuleType, SimpleNamespace

import pytest

from openwsp.backends.modal_runtime import configure_tls


class RecordingContext(ssl.SSLContext):
    def __new__(cls):
        return super().__new__(cls, ssl.PROTOCOL_TLS_CLIENT)

    def __init__(self):
        self.loaded = []

    def load_verify_locations(self, cafile=None, capath=None, cadata=None):
        self.loaded.append((cafile, capath, cadata))


@pytest.fixture
def modal_sdk(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    grpc_context = RecordingContext()
    grpc_calls, factory_calls = [], []

    class ModalChannel:
        def _get_default_ssl_context(self, **kwargs):
            grpc_calls.append(kwargs)
            return grpc_context

    class ProxyConnector:
        def __init__(self):
            self._ssl = RecordingContext()

    class ClientSessionRegistry:
        _client_session_active = False
        _client_session = None

    def original_factory(timeout):
        session = SimpleNamespace(connector=ProxyConnector(), timeout=timeout)
        factory_calls.append(session)
        return session

    modal = ModuleType("modal")
    modal.__path__ = []
    utils = ModuleType("modal._utils")
    utils.__path__ = []
    grpc = ModuleType("modal._utils.grpc_utils")
    grpc.ModalChannel = ModalChannel
    http = ModuleType("modal._utils.http_utils")
    http._http_client_with_tls = original_factory
    http.ClientSessionRegistry = ClientSessionRegistry
    modal._utils = utils
    utils.grpc_utils, utils.http_utils = grpc, http
    for module in (modal, utils, grpc, http):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(
        channel=ModalChannel, grpc_context=grpc_context, grpc_calls=grpc_calls,
        http=http, registry=ClientSessionRegistry, original_factory=original_factory,
        factory_calls=factory_calls, connector_type=ProxyConnector,
    )


def assert_verified(context):
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_ca_wraps_native_http_factory_without_replacing_proxy_or_timeout(modal_sdk, monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/trusted/environment.pem")
    configure_tls()
    assert not modal_sdk.factory_calls
    session = modal_sdk.http._http_client_with_tls(timeout=17.5)
    assert modal_sdk.factory_calls == [session]
    assert session.timeout == 17.5
    assert isinstance(session.connector, modal_sdk.connector_type)
    assert session.connector._ssl.loaded == [("/trusted/environment.pem", None, None)]
    assert_verified(session.connector._ssl)
    context = modal_sdk.channel()._get_default_ssl_context(test_option="preserved")
    assert context is modal_sdk.grpc_context
    assert modal_sdk.grpc_calls == [{"test_option": "preserved"}]
    assert context.loaded == [("/trusted/environment.pem", None, None)]
    assert_verified(context)


def test_repeated_configuration_does_not_stack_factories(modal_sdk, monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/trusted/environment.pem")
    configure_tls()
    http_factory = modal_sdk.http._http_client_with_tls
    grpc_factory = modal_sdk.channel._get_default_ssl_context
    configure_tls()
    assert modal_sdk.http._http_client_with_tls is http_factory
    assert modal_sdk.channel._get_default_ssl_context is grpc_factory
    session = http_factory(None)
    assert len(modal_sdk.factory_calls) == 1
    assert session.connector._ssl.loaded == [("/trusted/environment.pem", None, None)]


def test_existing_active_session_is_augmented_in_place(modal_sdk, monkeypatch):
    existing = modal_sdk.original_factory(60)
    connector, context = existing.connector, existing.connector._ssl
    modal_sdk.registry._client_session = existing
    modal_sdk.registry._client_session_active = True
    monkeypatch.setenv("SSL_CERT_FILE", "/trusted/environment.pem")
    configure_tls()
    assert modal_sdk.registry._client_session is existing
    assert existing.connector is connector and connector._ssl is context
    assert len(modal_sdk.factory_calls) == 1
    assert context.loaded == [("/trusted/environment.pem", None, None)]
    assert_verified(context)


def test_missing_ca_environment_leaves_sdk_untouched(modal_sdk):
    http_factory = modal_sdk.http._http_client_with_tls
    grpc_factory = modal_sdk.channel._get_default_ssl_context
    configure_tls()
    assert modal_sdk.http._http_client_with_tls is http_factory
    assert modal_sdk.channel._get_default_ssl_context is grpc_factory
    assert not modal_sdk.factory_calls and not modal_sdk.grpc_context.loaded


@pytest.mark.parametrize("primary", [None, "/trusted/primary.pem"])
def test_requests_bundle_fallback_and_ssl_bundle_precedence(modal_sdk, monkeypatch, primary):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/trusted/requests.pem")
    if primary is not None:
        monkeypatch.setenv("SSL_CERT_FILE", primary)
    configure_tls()
    expected = primary or "/trusted/requests.pem"
    session = modal_sdk.http._http_client_with_tls(timeout=None)
    context = modal_sdk.channel()._get_default_ssl_context()
    assert session.connector._ssl.loaded == [(expected, None, None)]
    assert context.loaded == [(expected, None, None)]


def test_previously_configured_grpc_does_not_skip_http_patch(modal_sdk, monkeypatch):
    modal_sdk.channel._openwsp_ca_configured = True
    grpc_factory = modal_sdk.channel._get_default_ssl_context
    monkeypatch.setenv("SSL_CERT_FILE", "/trusted/environment.pem")
    configure_tls()
    assert modal_sdk.channel._get_default_ssl_context is grpc_factory
    session = modal_sdk.http._http_client_with_tls(timeout=2)
    assert session.connector._ssl.loaded == [("/trusted/environment.pem", None, None)]
    assert_verified(session.connector._ssl)

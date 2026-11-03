"""Run the Modal CLI with configured certificate trust in this environment."""
from openwsp.backends.modal_runtime import configure_tls

configure_tls()
from modal.__main__ import main

if __name__ == "__main__":
    main()

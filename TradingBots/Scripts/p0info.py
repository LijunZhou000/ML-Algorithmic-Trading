from ib_insync import IB

from tradepy.config_old import IBConfig, RetryConfig
from tradepy.get_info.get_info import fetch_futures_dynamic
from tradepy.paths import FUTURES_STATIC, FUTURES_DYNAMIC


if __name__ == "__main__":

    ib_cfg = IBConfig()
    retry_cfg = RetryConfig()

    ib = IB()
    ib.connect(ib_cfg.host, ib_cfg.port, clientId=ib_cfg.client_id)

    try:
        fetch_futures_dynamic(
            ib=ib,
            static_path=FUTURES_STATIC,
            output_path=FUTURES_DYNAMIC,
            retry_config=retry_cfg,
        )
    finally:
        ib.disconnect()
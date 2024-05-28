import logging
import pytest

# Log Pytest Output
@pytest.fixture(scope="session", autouse=True)
def setup_logging(request):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('potholeClassifierLogger')

    def pytest_runtest_logreport(report):
        if report.when == "call":
            logger.info(report.nodeid + ": " + report.outcome.upper())
            for line in report.caplog:
                logger.info(line)

    def teardown_logging():
        logging.shutdown()

    request.addfinalizer(teardown_logging)
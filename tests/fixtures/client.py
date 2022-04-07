import pytest

from sc_api_tools import SCRESTClient
from sc_api_tools.http_session import SCSession


@pytest.fixture(scope="module")
def fxt_sc_session(fxt_vcr, fxt_server_config) -> SCSession:
    """
    This fixtures returns an SCSession instance which has already performed
    authentication
    """
    with fxt_vcr.use_cassette("session.yaml"):
        yield SCSession(cluster_config=fxt_server_config)


@pytest.fixture(scope="module")
def fxt_client(fxt_vcr, fxt_server_config) -> SCRESTClient:
    """
    This fixtures returns an SCRESTClient instance which has already performed
    authentication and retrieved a default workspace id
    """
    with fxt_vcr.use_cassette("client.yaml"):
        yield SCRESTClient(
            host=fxt_server_config.host,
            username=fxt_server_config.username,
            password=fxt_server_config.password
        )

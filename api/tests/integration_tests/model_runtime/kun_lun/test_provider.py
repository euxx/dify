import os

import pytest

from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.kun_lun.kun_lun import KunLunProvider
from tests.integration_tests.model_runtime.__mock.kun_lun import setup_kun_lun_mock


@pytest.mark.parametrize("setup_kun_lun_mock", [["none"]], indirect=True)
def test_validate_provider_credentials(setup_kun_lun_mock):
    provider = KunLunProvider()

    with pytest.raises(CredentialsValidateFailedError):
        provider.validate_provider_credentials(credentials={})

    provider.validate_provider_credentials(credentials={"kun_lun_api_key": os.environ.get("KUN_LUN_API_KEY")})

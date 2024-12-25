import dotenv

dotenv.load_dotenv(".env")

class ConfigKucoin(object):
    api_key = dotenv.get_key(".env", "KUCOIN_API_KEY")
    api_secret = dotenv.get_key(".env", "KUCOIN_API_SECRET")
    api_passphrase = dotenv.get_key(".env", "KUCOIN_API_PASSPHRASE")

    assert api_key and api_secret and api_passphrase 
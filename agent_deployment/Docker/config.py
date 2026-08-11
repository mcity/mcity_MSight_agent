import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    REGION = os.getenv('REGION', 'us-east-1')
    OPENAIAPIKEY = os.getenv('OPENAIAPIKEY')
    HFTOKEN = os.getenv('HFTOKEN')
    # cloudFormationManager.create_stack() reads these off self.config -- must stay defined.
    ANTHROPICAPIKEY = os.getenv('ANTHROPICAPIKEY', '')
    GEMINIAPIKEY = os.getenv('GEMINIAPIKEY', '')
    GROQAPIKEY = os.getenv('GROQAPIKEY', '')
    LSTOKEN = os.getenv('LSTOKEN', '')
    LSURL = os.getenv('LSURL', 'https://app.humansignal.com')
    CVATACCESSTOKEN = os.getenv('CVATACCESSTOKEN', '')
    ROLE_ARN = os.getenv('ROLE_ARN', 'arn:aws:iam::1234567800000:role/mcity-data-engine-agent-cf-role')
    KEY_PAIR_NAME = os.getenv('KEY_PAIR_NAME', 'mcity-data-engine-key')
    # Confirmed: r5 family, no GPU -- msight_docker.py auto-detects and runs MSight_Vision's CPU compose override.
    INSTANCE_TYPE = os.getenv('INSTANCE_TYPE', 'r5.large')
    STACK_NAME = "mcity-msight-agent"
    WHITELIST = set([
        "rpatnaik@umich.edu",
        "admin@umich.edu",
        "saisneha@umich.edu"
        # Add more whitelisted emails here.
    ])
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    MAX_USERS_PER_DAY = 20
    MAX_USER_REQUESTS_PER_DAY = 1
    STACK_DELETE_DELAY = 3600  # seconds
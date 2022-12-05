import os
from django.conf import settings
from django import setup

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'django.contrib.admin',
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.messages',
            'django.contrib.staticfiles',
            'core'
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'HOST': os.environ.get("DB_HOST"),
                'NAME': os.environ.get("DB_NAME"),
                'USER': os.environ.get("DB_USER"),
                'PASSWORD': os.environ.get("DB_PASSWORD")
            }
        }
    )
    setup()

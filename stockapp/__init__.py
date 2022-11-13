from flask import Flask
from flask_caching import Cache


config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}


app = Flask(__name__, instance_relative_config=False)
# tell Flask to use the above defined config
app.config.from_mapping(config)
cache = Cache(app)
cache.init_app(app, config={'CACHE_TYPE': 'SimpleCache'})

from stockapp import routes 


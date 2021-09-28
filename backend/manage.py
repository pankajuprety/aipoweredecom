#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import pickle

from keras import models
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def importClasses():
    with open('classes.pkl','rb') as modelFile:
        classes = pickle.load(modelFile)
    return classes

def importWords():
    with open('words.pkl','rb') as modelFile:
        words = pickle.load(modelFile)
    return words

def importModel():
    model = models.load_model('chatbot_model.h5')
    return model


def main():
    """Run administrative tasks."""
    classes = importClasses()
    words = importWords()
    model = importModel()
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

#!/bin/bash
# Starts the three parts of the bot inside one container.
# HuggingFace Spaces exposes a single port (7860); Rasa (5005) and the action server (5055) stay private.
rasa run actions --port 5055 &
rasa run --port 5005 --model models/eco-travel.tar.gz --endpoints endpoints.yml --credentials credentials.yml &
python serve.py
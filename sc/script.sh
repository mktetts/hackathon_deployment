#!/bin/sh

truffle compile

# python3 deploy.py

mv build/contracts/* ../python_backend/contract/
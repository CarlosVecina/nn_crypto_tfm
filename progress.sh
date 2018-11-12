#!/usr/bin/env bash
watch -d "ls -lht ../s3/data/gdax/historic | head -n 4; supervisorctl tail -126 gdax_saver; supervisorctl tail -126 gdax_saverb;supervisorctl status; ls -lt ../s3/data/gdax/historic | head -n 4"

#!/bin/bash

URL=http://localhost:8000/compare 
STATUS_CODE=$(curl -X POST -H "Content-Type: application/octet-stream" --data-binary $'\x82\xa6prompt\xa2Hi\xaacandidates\x92\xa6Hello!\xa4Hey!' -o /dev/null -s -w "%{http_code}" "$URL")

if [ "$STATUS_CODE" -eq 200 ]; then
  echo "Success"
  exit 0  # success
else
  echo "Failure"
  exit 1  # failure
fi
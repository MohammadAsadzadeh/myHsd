# HateSpeechDetection
Persian hate speech detection, for master of science proposal.  
## Installation
- clone the project
- set environment variables in docker-compose.yml file
- run `docker compose up`

## Environment variables
**DB_PASS**: root password to connect to mysql. it should be same as **MYSQL_ROOT_PASSWORD**  
**ADMIN_USERNAME**: username to login to admin of app  
**ADMIN_PASSWORD**: password to login ti admin of app  
**SECRET_KEY**: secret key to encode jwt data  
**MYSQL_ROOT_PASSWORD**: mysql root password

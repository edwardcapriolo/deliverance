docker run -p 8085:8080 -it -v ~/.deliverance:/home/reckoning/.deliverance:ro -e RECKONING_OPTS=" -Dspring.config.location=file:/reckoning/simple.yaml " ecapriolo/reckoning:0.0.9

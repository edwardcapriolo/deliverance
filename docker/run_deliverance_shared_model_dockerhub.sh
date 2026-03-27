docker run -p 8085:8080 -it -v ~/.deliverance:/home/deliverance/.deliverance:ro -e DELIVERANCE_OPTS=" -Dspring.config.location=file:/deliverance/simple.properties " ecapriolo/deliverance:0.0.5

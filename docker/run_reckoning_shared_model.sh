docker run -p 8085:8080 -it --mount type=bind,src=.deliverance,dst=/home/reckoning/.deliverance -e RECKONING_OPTS=" -Dspring.config.location=file:/reckoning/simple.yaml " reckoning

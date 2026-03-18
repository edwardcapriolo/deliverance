FROM ecapriolo/jdk-25:0.0.1 AS deliverance-base
RUN apk add git
RUN apk add maven

#native module
RUN apk add curl jq unzip bash make clang

RUN mkdir /build
WORKDIR /build
RUN cd /build
RUN git clone https://github.com/edwardcapriolo/deliverance.git

RUN cd /build/deliverance
WORKDIR /build/deliverance
RUN --mount=type=cache,target=/root/.m2 cd /build/deliverance && mvn install -Dmaven.test.skip.exec=true -Dgpg.skip=true

FROM ecapriolo/jdk-25:0.0.1 AS deliverance
RUN mkdir /deliverance
RUN apk add bash

RUN addgroup -S deliverance && adduser -S -G deliverance -H -D deliverance
RUN mkdir /deliverance/logs && chown deliverance:deliverance /deliverance/logs
COPY --from=deliverance-base /build/deliverance/web/target/web-0.0.4-SNAPSHOT.jar /deliverance/web.jar
COPY entry_point.sh /deliverance/entry_point.sh
COPY inlinerules.json /deliverance/inlinerules.json
COPY simple.properties /deliverance/

RUN chmod u+x /deliverance/entry_point.sh
WORKDIR /deliverance
USER deliverance
#ENTRYPOINT ["java", "--add-modules", "jdk.incubator.vector", "--add-opens", "java.base/java.nio=ALL-UNNAMED", "-Xmx2G", "-jar", "web.jar"] 
ENTRYPOINT ["/deliverance/entry_point.sh"] 

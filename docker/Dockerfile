##################STAGE1#################
FROM ecapriolo/jdk-25:0.0.2 AS deliverance-base
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
#RUN --mount=type=cache,target=/root/.m2 cd /build/deliverance && mvn install -Dmaven.test.skip.exec=true -Dgpg.skip=true
RUN --mount=type=cache,target=/root/.m2 cd /build/deliverance && mvn dependency:go-offline

##################STAGE2#################
FROM deliverance-base AS deliverance-sha
ARG REPO_COMMIT_SHA=LATEST
RUN cd /build/deliverance
RUN git checkout 
RUN --mount=type=cache,target=/root/.m2 cd /build/deliverance && mvn install -Dmaven.test.skip.exec=true -Dgpg.skip=true


##################STAGE3#################
FROM ecapriolo/jdk-25:0.0.2 AS deliverance
RUN mkdir /deliverance
RUN apk add bash

RUN addgroup -S deliverance && adduser -S -G deliverance -H -D deliverance
RUN mkdir /deliverance/logs && chown deliverance:deliverance /deliverance/logs
COPY --from=deliverance-sha /build/deliverance/web/target/web-0.0.10-SNAPSHOT.jar /deliverance/web.jar
COPY entry_point.sh /deliverance/entry_point.sh
COPY inlinerules.json /deliverance/inlinerules.json
COPY simple.properties /deliverance/

RUN chmod u+x /deliverance/entry_point.sh
WORKDIR /deliverance
USER deliverance
ENTRYPOINT ["/deliverance/entry_point.sh"]

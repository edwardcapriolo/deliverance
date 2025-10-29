export JAVA_HOME=/usr/lib/jvm/java-24-temurin-jdk
mvn spring-boot:run -Dspring-boot.run.jvmArguments="--add-modules jdk.incubator.vector -Xmx6G"
#java --add-modules jdk.incubator.vector -Xmx6G -cp target/web-0.0.1-SNAPSHOT.jar com.deliverance.http.DeliveranceApplication

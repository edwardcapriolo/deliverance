# Vibrant Maven Plugin

`vibrant-maven-plugin` runs spec-driven code generation from Maven. Put a `vibeSpec` in your project POM, then run the plugin to generate code from the configured system and user messages.

## Example Configuration

```xml
<plugin>
    <groupId>io.teknek.deliverance</groupId>
    <artifactId>vibrant-maven-plugin</artifactId>
    <!--<version>0.0.4</version> -->
    <configuration>
        <vibeSpecs>
            <vibeSpec>
                <id>shape</id>
                <enabled>true</enabled>
                <overwrite>true</overwrite>
                <systemMessages>
                    <systemMessage>You are an assistant that produces concise, production-grade software.</systemMessage>
                    <systemMessage>Output java code.</systemMessage>
                    <systemMessage>Generate java code into the package 'io.teknek.shape'.</systemMessage>
                </systemMessages>
                <userMessages>
                    <userMessage>Generate a java interface named Shape with a method named area that returns a double.</userMessage>
                    <userMessage>Generate a java class named Circle that implements the Shape interface.</userMessage>
                </userMessages>
                <generateTo>generated-source</generateTo>
            </vibeSpec>
        </vibeSpecs>
    </configuration>
</plugin>
```

## Run

```sh
export JAVA_HOME=/usr/lib/jvm/java-25-temurin-jdk/
export MAVEN_OPTS="--add-opens java.base/java.nio=ALL-UNNAMED --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Djava.library.path=/home/edward/deliverence/native/target/native-lib-only"
mvn io.teknek.deliverance:vibrant-maven-plugin:0.0.4-SNAPSHOT:generate
```

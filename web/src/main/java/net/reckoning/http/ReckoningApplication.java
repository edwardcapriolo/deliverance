package net.reckoning.http;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

import java.util.ArrayList;

@SpringBootApplication()
@ComponentScan(basePackages = {"net.reckoning"})
public class ReckoningApplication {

    public static void main(String[] args) {
        SpringApplication.run(ReckoningApplication.class, args);
    }
}

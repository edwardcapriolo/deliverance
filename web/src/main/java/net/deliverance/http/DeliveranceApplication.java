package net.deliverance.http;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

import java.util.ArrayList;

@SpringBootApplication()
@ComponentScan(basePackages = {"net.deliverance.http"})
public class DeliveranceApplication {

    public static void main(String[] args) {

        SpringApplication.run(DeliveranceApplication.class, args);
    }
}

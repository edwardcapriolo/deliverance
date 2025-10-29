package net.deliverance.http;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.env.Environment;
import org.springframework.test.context.junit.jupiter.SpringExtension;

@ExtendWith(SpringExtension.class)
@SpringBootTest(args = "--add-modules jdk.incubator.vector")
public class DeliveranceControllerTest {

    @Test
    public void whenUsingSpringBootTestArgs_thenCommandLineArgSet(@Autowired Environment env) {
        //Assertions.assertThat(env.getProperty("spring.main.banner-mode")).isEqualTo("off");
    }
}
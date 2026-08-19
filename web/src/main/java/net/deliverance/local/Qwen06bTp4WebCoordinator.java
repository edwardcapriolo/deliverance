package net.deliverance.local;

import net.deliverance.http.Config;
import net.deliverance.http.DeliveranceApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;

public class Qwen06bTp4WebCoordinator {
    public static void main(String[] args) {
        Config.useLocalLauncherPoolSize(LocalQwen06bTp4.POOL_SIZE);
        new SpringApplicationBuilder(DeliveranceApplication.class)
                .properties(LocalQwen06bTp4.coordinatorProperties())
                .run(args);
    }
}

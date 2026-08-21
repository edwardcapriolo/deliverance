package net.deliverance.local;

import net.deliverance.http.DeliveranceApplication;

public final class Qwen06bTp4WebCoordinator {
    private Qwen06bTp4WebCoordinator() {
    }

    public static void main(String[] args) {
        DeliveranceApplication.main(LocalQwen06bTp4.webArgs(args));
    }
}

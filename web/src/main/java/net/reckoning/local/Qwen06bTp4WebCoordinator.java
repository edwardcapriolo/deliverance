package net.reckoning.local;

import net.reckoning.http.ReckoningApplication;

public final class Qwen06bTp4WebCoordinator {
    private Qwen06bTp4WebCoordinator() {
    }

    public static void main(String[] args) {
        ReckoningApplication.main(LocalQwen06bTp4.webArgs(args));
    }
}

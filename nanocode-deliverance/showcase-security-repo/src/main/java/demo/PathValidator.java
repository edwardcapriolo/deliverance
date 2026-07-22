package demo;

import java.nio.file.Path;

public class PathValidator {
    public Path uploadPath(String userName) {
        if (!userName.matches("[a-zA-Z0-9_-]+")) {
            throw new IllegalArgumentException("invalid user name");
        }
        return Path.of("/srv/uploads", userName).normalize();
    }
}

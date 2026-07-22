package demo;

import java.io.IOException;

public class ArchiveController {
    public Process createArchive(String userName) throws IOException {
        String command = "tar -czf /tmp/" + userName + ".tgz /srv/uploads/" + userName;
        return Runtime.getRuntime().exec(command);
    }
}


package com.github.tjake.jlama.tensor.operations.util;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;

import io.teknek.deliverance.tensor.operations.RuntimeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JarSupport {
    private static final Logger LOGGER = LoggerFactory.getLogger(JarSupport.class);

    public static boolean maybeLoadLibrary(String libname) {
        String ext = RuntimeSupport.isMac() ? ".dylib" : RuntimeSupport.isWin() ? ".dll" : ".so";
        String name = "lib" + libname + ext;
        URL lib = JarSupport.class.getClassLoader().getResource("META-INF/native/lib/" + name);

        if (lib == null) {
            name = libname + ext;
            lib = JarSupport.class.getClassLoader().getResource("META-INF/native/lib/" + name);
        }

        if (lib != null) {
            try {
                final File libpath = Files.createTempDirectory("deliverance").toFile();
                libpath.deleteOnExit(); // just in case

                File libfile = Paths.get(libpath.getAbsolutePath(), name).toFile();
                libfile.deleteOnExit(); // just in case

                final InputStream in = lib.openStream();
                final OutputStream out = new BufferedOutputStream(new FileOutputStream(libfile));

                int len;
                byte[] buffer = new byte[8192];
                while ((len = in.read(buffer)) > -1) {
                    out.write(buffer, 0, len);
                }
                out.close();
                in.close();
                System.load(libfile.getAbsolutePath());
                LOGGER.debug("Loaded {} library: {}", libname, libfile.getAbsolutePath());
                return true;
            } catch (IOException e) {
                LOGGER.warn("Error loading {} library", libname);
            }
        }

        LOGGER.warn("jlama-native shared library not found: {}{}", libname, ext);
        return false;
    }
}

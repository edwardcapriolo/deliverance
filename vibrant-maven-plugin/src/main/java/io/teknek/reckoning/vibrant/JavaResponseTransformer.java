package io.teknek.reckoning.vibrant;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;


//this is hacky without structured output
//we assume one package lots of imports and multiple classes /interfaces
// in one reply.
public class JavaResponseTransformer implements MessageTransformer {
    public void transform(String raw, Path base, VibeSpec vibeSpec){
        String currentPackage = null;
        String trimMd = processResponse(raw);
        BufferedReader br = new BufferedReader(new StringReader(trimMd));
        String line;
        List<String> imports = new ArrayList<>();
        try {
            while ((line = br.readLine()) != null) {
                if (line.contains("package ")){
                    currentPackage = line.split("\\s+")[1];
                    currentPackage = currentPackage.replace(";","");
                }
                if (line.contains("import ")){
                    imports.add(line);
                }
            }
        } catch (IOException e){
            throw new UncheckedIOException(e);
        }
        if (currentPackage != null){
            base = makePackageFolders(base, currentPackage);
        }

        String currentFile = null;
        StringBuilder file = new StringBuilder();
        //if (currentPackage != null){
        //    file.append("package " + currentPackage + "\n");
        //}
        //for (String inport: imports){
        //    file.append(inport + "\n");
        //}
        //2nd pass
        br = new BufferedReader(new StringReader(trimMd));
        try {
            while ((line = br.readLine()) != null) {
                if (line.startsWith("import ")){
                    continue;
                }
                if (line.startsWith("package ")){
                    continue;
                }
                if (line.contains("public class")){
                    currentFile = line.split("\\s+")[2];
                    if (currentPackage != null){
                        file.append("package "+ currentPackage +";\n");
                    }
                    for (String impor: imports){
                        file.append(impor + "\n");
                    }
                }
                if (line.contains("public interface")){
                    currentFile = line.split("\\s+")[2];
                    if (currentPackage != null){
                        file.append("package "+ currentPackage +";\n");
                    }
                    for (String inport: imports){
                        file.append(inport + "\n");
                    }
                }
                file.append(line + "\n");
                if (line.length() >= 1 && line.charAt(0) == '}'){
                    if (currentFile != null) {
                        Path p = base.resolve(currentFile + ".java");
                        System.out.println("writing to path "+ p);
                        if (vibeSpec.isOverwrite()) {
                            Files.write(p, file.toString().getBytes(StandardCharsets.UTF_8));
                        } else {
                            Files.write(p, file.toString().getBytes(StandardCharsets.UTF_8),  StandardOpenOption.CREATE_NEW);
                        }
                        file = new StringBuilder();
                    }
                }
            }
        } catch (IOException e){
            throw new UncheckedIOException(e);
        }

    }

    public static String processResponse(String input){
        String s = "```java";
        int indexOfStart = input.indexOf(s);
        if (indexOfStart == -1){
            //assume not MD all code
            return input;
        }
        int end = input.lastIndexOf("```");
        if (end == -1){
            end = input.length() -1;
        }
        return input.substring(indexOfStart + s.length() + 1, end );
    }

    public static Path makePackageFolders(Path base, String pack){
        File f = new File(base.toFile().getAbsolutePath(), pack.replace('.', '/'));
        boolean ignore = f.mkdirs();
        return f.toPath();
    }
}

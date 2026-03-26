Run the tests. Check "git tag -l"
  ```
  mvn versions:set -DnewVersion=0.0.5
	mvn deploy -Dmaven.test.skip.exec=true
	git commit -a -m "Release version update"
 	git tag v0.0.5
 	git push origin v0.0.5
  mvn versions:set -DnewVersion=0.0.6-SNAPSHOT
	git commit -a -m "Return to snapshot"
	git push origin main
  ```

  The sonatype web ui:

  https://central.sonatype.com/publishing/deployments

  The release should be validated. Push the publish button.

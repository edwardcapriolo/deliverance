# Disk KV Backend Progress

## Timebox

* Start: 2026-06-02T04:30:54+00:00
* Budget: up to two hours
* Scope: active `KvBuffer` disk-page storage backend, not persistent prefix-cache storage

## Initial Findings

* `KvBufferCacheSettings(File)` enables disk-backed active KV pages under a caller-provided directory.
* Disk pages are currently flat files named `<session>-L<layerPage>C<contextPage>.page`.
* The active disk backend has no durable manifest, token index, model fingerprint, or prefix-key lookup. Those are prefix-cache concerns and are intentionally out of scope here.
* Range reads in `getTensorsUptoPosition(...)` create missing pages with the upper-bound context page id instead of the loop page id. That can alias multiple logical context pages onto one physical disk page.
* Existing disk pages are closed but not deleted, so active KV pages can accumulate in the working directory.

## Planned Changes

* Fix range-read disk page naming so each logical page maps to the expected file.
* Ensure the disk working directory exists before page allocation.
* Add active-page cleanup on close by default for disk-backed buffers.
* Add settings for explicitly keeping disk pages after close for debugging/performance inspection.
* Add Dropwizard metrics for disk page creation/open/close/delete/error and byte accounting.
* Add focused tests for page naming, cleanup, retained pages, and metrics.
* Add documentation separating active disk-page storage from persistent prefix caching.

## Checkpoints

* 2026-06-02T04:30:54+00:00: Started timebox and inspected implementation/tests/docs.
* Added focused `DiskKvBackendTest` tests for disk page naming, default cleanup, retained pages, and metrics.
* Implemented working-directory creation, active-page delete-on-close, retention setting, disk lifecycle metrics, and the range-read page-id fix.
* Full x86 `mvn clean compile` was attempted but was too slow under emulation and was stopped. Verification should be rerun on the ARM environment.
* Added daemon stale-page sweeper for disk-backed caches with settings for enablement, sweep interval, and max page age.
* Added active-page tracking so the sweeper skips pages still open by the current cache instance.
* Added deterministic sweeper tests to `DiskKvBackendTest`; tests were not run per user instruction.
* Added `AbstractModel.runDiskKvPageSweep()` / `KvBufferCache.runDiskPageSweep()` maintenance hook for forced sweeps.
* Added Gemma integration coverage that generates with disk-backed active KV pages, verifies page files exist, ages them, forces a sweep, and verifies cleanup.

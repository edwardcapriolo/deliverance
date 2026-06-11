# Disk KV Backend

The disk KV backend is an active storage backend for live `KvBufferCache.KvBuffer` pages. It lets a live generation use
memory-mapped page files instead of allocating every KV page through the tensor allocator.

This is not a persistent prefix cache. Disk page files are named by active buffer session and page coordinates, not by
token prefixes, and there is no durable token index, manifest, model fingerprint, or cross-process reuse contract.

Disk KV also does not store in-memory prefix-cache snapshots. When `KvBufferCacheSettings(File)` is used, prefix-cache
storage is skipped and only the live request's active KV pages are backed by disk. This prevents the snapshot-copy prefix
cache from creating many cumulative disk-backed KV buffers for every block-aligned prompt prefix.

## Configuration

Use `KvBufferCacheSettings(File workingDirectory)` to enable disk-backed active pages:

```java
KvBufferCacheSettings settings = new KvBufferCacheSettings(new File("/tmp/deliverance-kv"));
```

The backend creates the working directory if it does not already exist.

By default, page files are deleted when the owning `KvBuffer` closes. This matches the active-page storage contract and
prevents stale generation pages from accumulating indefinitely. Disk KV should be treated as an opt-in memory-pressure
fallback, not the default fast path; it trades direct-memory pressure for memory-mapped file I/O.

For debugging or performance inspection, pages can be retained:

```java
KvBufferCacheSettings settings = new KvBufferCacheSettings(new File("/tmp/deliverance-kv"))
        .withDeleteDiskPagesOnClose(false);
```

Retained files are still not reusable prefix-cache entries. They are raw active page files and should be deleted by the
operator or test harness after inspection.

The backend also starts a daemon sweeper by default. The sweeper removes stale `.page` files in the working directory
that are older than `diskPageMaxAge` and are not currently open by this `KvBufferCache` instance. This is a safety net
for orphaned files from crashes, interrupted runs, or retained inspection pages; normal cleanup still happens when
`KvBuffer.close()` deletes active pages.

Sweeper settings:

```java
KvBufferCacheSettings settings = new KvBufferCacheSettings(new File("/tmp/deliverance-kv"))
        .withDiskPageSweeperEnabled(true)
        .withDiskPageSweepInterval(Duration.ofMinutes(5))
        .withDiskPageMaxAge(Duration.ofHours(1));
```

The sweeper is intentionally age-based. It should not delete currently open pages from the same cache instance, but it is
not a cross-process lock manager. Avoid sharing one disk KV directory across unrelated running processes unless stale-file
cleanup is acceptable for that deployment.

## Page Names

Disk files use this layout:

```text
<session>-L<layerPage>C<contextPage>.page
```

For example:

```text
request-123-L0C0.page
request-123-L0C1.page
```

Each logical layer/context page must map to its own physical file. Range reads through `getKeyTensorsUptoPosition(...)`
or `getValTensorsUptoPosition(...)` allocate any missing earlier context pages using the page's own context index.

## Metrics

The disk backend records these Dropwizard metrics:

* `kvbuffercache.disk.directory.error`: working-directory creation or validation failed.
* `kvbuffercache.disk.page.open`: a disk page file was opened.
* `kvbuffercache.disk.page.open.error`: a disk page file could not be opened or mapped.
* `kvbuffercache.disk.page.create`: a new disk page file was created.
* `kvbuffercache.disk.page.close`: a disk-backed page object closed successfully.
* `kvbuffercache.disk.page.close.error`: closing a disk-backed page failed.
* `kvbuffercache.disk.page.delete`: a close deleted a disk page file.
* `kvbuffercache.disk.page.delete.error`: deleting a disk page file failed.
* `kvbuffercache.disk.bytes.allocated`: cumulative bytes allocated for newly created disk pages.
* `kvbuffercache.disk.bytes.deleted`: cumulative bytes deleted from disk page files.
* `kvbuffercache.disk.bytes.live`: bytes created by this cache instance that remain live on disk.
* `kvbuffercache.disk.sweeper.run`: a disk sweeper pass started.
* `kvbuffercache.disk.sweeper.error`: a disk sweeper pass failed.
* `kvbuffercache.disk.sweeper.page.scan`: a `.page` file was inspected by the sweeper.
* `kvbuffercache.disk.sweeper.page.delete`: a stale `.page` file was deleted by the sweeper.
* `kvbuffercache.disk.sweeper.page.skip.active`: a `.page` file was skipped because this cache still has it open.
* `kvbuffercache.disk.sweeper.page.skip.young`: a `.page` file was skipped because it is newer than `diskPageMaxAge`.
* `kvbuffercache.prefix.disk.skip`: prefix snapshot storage was skipped because disk KV is active.

`kvbuffercache.disk.bytes.live` is process-local accounting for pages created by the current cache instance. It is not a
startup scanner for files retained by a prior run.

## Prefix Cache Boundary

A persistent prefix cache would need a separate durable index keyed by token-prefix hashes plus model/config/tokenizer
fingerprints. This backend intentionally does not provide that. It only stores active KV pages for the lifetime of their
owning buffer, and prefix snapshot storage is disabled in disk-backed mode.

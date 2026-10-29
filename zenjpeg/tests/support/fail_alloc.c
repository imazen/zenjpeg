/* Linux-only test subprocess: fail one allocation of an exact requested size.
 * The normal library and test process never load this interposer. */
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdatomic.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
static atomic_int failed;
static int reject(size_t n) {
  const char *wanted=getenv("ZENJPEG_FAIL_ALLOC_SIZE");
  if (wanted && n==strtoul(wanted,NULL,10) && !atomic_exchange(&failed,1)) {
    errno=ENOMEM; return 1;
  }
  return 0;
}
void *malloc(size_t n) { return reject(n) ? NULL : __libc_malloc(n); }
void *calloc(size_t n,size_t size) {
  return size && n>((size_t)-1)/size ? NULL : reject(n*size) ? NULL : __libc_calloc(n,size);
}
void *realloc(void *p,size_t n) { return reject(n) ? NULL : __libc_realloc(p,n); }

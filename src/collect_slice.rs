// Vendored from https://crates.io/crates/collect_slice
pub fn collect_slice<T>(iter: T, slice: &mut [T::Item]) -> usize
where
    T: Iterator,
{
    slice.iter_mut().zip(iter).fold(0, |count, (dest, item)| {
        *dest = item;
        count + 1
    })
}

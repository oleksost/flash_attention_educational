# Flash Attention Educational
Implementation of flash attention for educational purposes. This does not implement dropout and masking to keep it simple.

This is under construction.

This will include:

- [x] implemenation in numba
    - [x] forward
    - [x] backward
- [ ] implementation in C++
    - [X] forward
    - [ ] backward
- [ ] implementation in triton
- [ ] Profile + optimize



_______
### Known issues
- [ ] for numba implementation backward grads for Q and K do not align with standard attention grads

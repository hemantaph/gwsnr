Test the functionality of the package with pytest.
Run pytest from the root directory of the package to execute all tests.

```
tests/
├── unit/
│   ├── test_GWSNR_interpolation.py   
│   ├── test_GWSNR_inner_product.py   
│   ├── test_GWSNR_inner_product_jax.py           
│   └── test_ann.py             
├── integration/
│   ├── test_SNR_BBH_non_spinning.py
│   ├── test_SNR_BBH_spin_precession.py
│   ├── test_Pdet_BBH_spin_precession.py
```
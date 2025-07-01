
flowchart LR
 subgraph MAIN["🌟 GWSNR Package"]
        GWSNR_CLASS["🏢 GWSNR Class (Main Interface)"]
  end
 subgraph SNR["📊 SNR Calculation Methods"]
        INTERPOLATION["🔄 Interpolation"]
        INNER_PRODUCT["🔢 Inner Product"]
        ANN_METHOD["🧠 ANN"]
        HYBRID["🔀 Hybrid Strategy"]
  end
 subgraph INTERP_SECTION["Interpolation"]
    direction TB
        INTERP_DETAIL["🔄 Details (2D/4D, Caching)"]
        NUMBA["🚀 Numba Backend"]
        JAX["⚡ JAX Backend"]
  end
 subgraph INNER_PROD_SECTION["Inner Product"]
    direction TB
        INNER_DETAIL["🔢 Details (Bilby/JAX Style)"]
        RIPPLE["🌊 Ripple Module (Waveforms)"]
        MULTIPROC["⚡ Multiprocessing"]
  end
 subgraph ANN_SECTION["ANN"]
    direction TB
        ANN["🧠 ANN Module (P_det)"]
  end
 subgraph HYBRID_SECTION["Hybrid"]
    direction TB
        INTERPOLATION_REF["Ref: Interpolation or ANN"]
        INNER_PRODUCT_REF["Ref: Inner Product"]
  end
 subgraph DETAILS["⚙️ Method Details & Implementations"]
    direction TB
        INTERP_SECTION
        INNER_PROD_SECTION
        ANN_SECTION
        HYBRID_SECTION
  end
 subgraph SUPPORT["🛠️ Supporting Infrastructure"]
    direction LR
        UTILS["🔧 Utils"]
        FEATURES["✨ Key Features"]
        PERF["⚡ Performance Opts"]
        EXTERNAL["🔗 Dependencies"]
  end
    GWSNR_CLASS --> SNR
    INTERPOLATION --> INTERP_DETAIL
    INTERP_DETAIL --> NUMBA & JAX
    INNER_PRODUCT --> INNER_DETAIL
    INNER_DETAIL --> RIPPLE & MULTIPROC
    ANN_METHOD --> ANN
    HYBRID --> INTERPOLATION_REF & INNER_PRODUCT_REF
    GWSNR_CLASS -.-> SUPPORT
    INTERP_DETAIL@{ shape: rect}

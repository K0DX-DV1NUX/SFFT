# SFFT 

## Overview




## Key Features


## Usage

### Initialization

```
model = Model(configs)
```
### Configuration Options
| Parameter       | Description                                    | Default |
|---------------|--------------------------------|---------|
| `seq_len`     | Input sequence length                         | -       |
| `pred_len`    | Output prediction length                      | -       |
| `enc_in`    | Number of input features                      | -       |
| `individual`  | If True, applies a separate layer per feature | False   |
| `bias`        | Enables bias in the low-rank layer            | True    |
| `enable_Haar` | Enables Haar decomposition                    | True    |
| `enable_DCT`  | Enables Discrete Cosine Transform             | True    |
| `enable_iDCT`  | Enables Inverse Discrete Cosine Transform    | False    |
| `enable_lowrank`  | Enables Low Rank or Standard Linear Layer    | True    |
| `rank`        | Rank of the low-rank layer                    | 30      |

### Forward Pass


# Results




# Acknowledgement
The listed github repositories have been used for code bases, datasets and comparisons.

- https://github.com/zhouhaoyi/Informer2020
- https://github.com/lss-1138/SparseTSF
- https://github.com/cure-lab/LTSF-Linear
- https://github.com/aikunyi/FreTS
- https://github.com/luodhhh/ModernTCN
- https://github.com/yuqinie98/patchtst
- https://github.com/thuml/iTransformer
- https://github.com/SiriZhang45/FRNet


## License
This project is licensed under the MIT License.

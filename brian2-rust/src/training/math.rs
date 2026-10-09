//! Standard scalar math with explicit derivatives and stable removable limits.
use serde::{Deserialize, Serialize};

// Appended GPU opcode 52 carries this discriminant in its parameter word.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u64)]
pub enum Function {
    Tan, Cosh, Sinh, Log10, Expm1, Log1p, Exprel,
    Arccos, Arcsin, Arctan, Ceil, Abs, Sign, Floor,
}

impl Function {
    pub fn value(self, x: f64) -> f64 {
        match self {
            Self::Tan => x.tan(), Self::Cosh => x.cosh(), Self::Sinh => x.sinh(),
            Self::Log10 => x.log10(), Self::Expm1 => x.exp_m1(), Self::Log1p => x.ln_1p(),
            Self::Exprel => if x == 0.0 { 1.0 } else { x.exp_m1()/x },
            Self::Arccos => x.acos(), Self::Arcsin => x.asin(), Self::Arctan => x.atan(),
            Self::Floor => x.floor(), Self::Ceil => x.ceil(), Self::Abs => x.abs(),
            Self::Sign => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        }
    }
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Tan => 1.0/x.cos().powi(2), Self::Cosh => x.sinh(), Self::Sinh => x.cosh(),
            Self::Log10 => if x>1.0 { (1.0/x)/std::f64::consts::LN_10 } else { 1.0/(x*std::f64::consts::LN_10) }, Self::Expm1 => x.exp(),
            Self::Log1p => 1.0/(1.0+x),
            Self::Exprel => if x.abs()<0.01 {
                0.5+x*(1.0/3.0+x*(1.0/8.0+x*(1.0/30.0+x*(1.0/144.0+x*(1.0/840.0+x*(1.0/5760.0+x/45360.0))))))
            } else if x < -50.0 { (1.0/x).powi(2) }
            else if x > 50.0 { (x.exp_m1()*(1.0-1.0/x)+1.0)/x }
            else { ((x-1.0)*x.exp_m1()+x)/(x*x) },
            Self::Arccos => -1.0/((1.0-x)*(1.0+x)).sqrt(),
            Self::Arcsin => 1.0/((1.0-x)*(1.0+x)).sqrt(),
            Self::Arctan => if x.abs()>1.0 { let inv=1.0/x; inv*inv/(1.0+inv*inv) } else { 1.0/(1.0+x*x) },
            Self::Abs => if x>0.0 {1.0} else if x<0.0 {-1.0} else {0.0},
            Self::Ceil | Self::Sign | Self::Floor => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Function::*;
    #[test]
    fn removable_limit_and_small_values() {
        assert_eq!(Exprel.value(0.0),1.0);
        assert_eq!(Exprel.derivative(0.0),0.5);
        for x in [-1e-12,1e-12,-1e-8,1e-8] {
            assert!((Expm1.value(x)/x-1.0).abs()<1e-8);
            assert!((Log1p.value(x)/x-1.0).abs()<1e-8);
            assert!((Exprel.derivative(x)-(0.5+x/3.0+x*x/8.0)).abs()<1e-16);
        }
        assert_eq!(Abs.derivative(-0.0),0.0);
        assert_eq!(Sign.value(-0.0),0.0);
    }
    #[test]
    fn interior_derivatives() {
        for f in [Tan,Cosh,Sinh,Log10,Expm1,Log1p,Exprel,Arccos,Arcsin,Arctan,Ceil,Abs,Sign,Floor] {
            for x in [0.13,0.35,0.78] {
                let fd=(f.value(x+1e-6)-f.value(x-1e-6))/2e-6;
                assert!((fd-f.derivative(x)).abs()<2e-8);
            }
        }
    }
}

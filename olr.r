#Data loading and cleaning
Further_education_ord <- read.csv("C:/Users/User/Downloads/Fordinal.csv")
View(Further_education_ord)

str(Further_education_ord)
Further_education_ord$cpa.satisfaction = as.factor(Further_education_ord$cpa.satisfaction)
Further_education_ord$confidence.for.job = as.factor(Further_education_ord$confidence.for.job)
Further_education_ord$further.edu = as.factor(Further_education_ord$further.edu)
str(Further_education_ord)

nrow(Further_education_ord[is.na(Further_education_ord)])
xtabs(~further.edu + sex , data=Further_education_ord) #keep
xtabs(~further.edu + family.members, data=Further_education_ord) #reject
xtabs(~further.edu + dependents, data=Further_education_ord) #keep
xtabs(~further.edu + house.income, data=Further_education_ord) #reject
xtabs(~further.edu + fm.futher.edu, data=Further_education_ord) #keep
xtabs(~further.edu + aspiration, data=Further_education_ord) #keep
xtabs(~further.edu + move.countries, data=Further_education_ord)#keep
xtabs(~further.edu + cpa, data=Further_education_ord) #reject
xtabs(~further.edu + cpa.satisfaction, data=Further_education_ord) #reject
xtabs(~further.edu + star.family, data=Further_education_ord) #keep
xtabs(~further.edu + job.expectation, data=Further_education_ord) #accept
xtabs(~further.edu + confidence.for.job, data=Further_education_ord) #reject

#Model building 
require(foreign)
require(reshape2)
require(Hmisc)
require(MASS)

olr1 <- polr(further.edu ~ sex, data = Further_education_ord, Hess=TRUE)
summary(olr1)
ctable <- coef(summary(olr1))
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
ctable <- cbind(ctable, "p value" = p)
ctable   #not sig
#odds ratio
exp(coef(olr1)) 


#complete model
olrcomplete <- polr(further.edu ~ sex + dependents + aspiration + move.countries + fm.futher.edu + star.family +job.expectation, data = Further_education_ord, Hess=TRUE)
summary(olrcomplete)
ctable2 <- coef(summary(olrcomplete))
p <- pnorm(abs(ctable2[, "t value"]), lower.tail = FALSE) * 2
ctable2 <- cbind(ctable2, "p value" = p)
ctable2

#multicolinearity check
library(car)
vif(olrcomplete)

#revised model
olrrevised <- polr(further.edu ~ aspiration, data = Further_education_ord, Hess=TRUE)
summary(olrrevised)
ctable3 <- coef(summary(olrrevised))
p <- pnorm(abs(ctable3[, "t value"]), lower.tail = FALSE) * 2
ctable3 <- cbind(ctable3, "p value" = p)
ctable3
exp(coef(olrrevised))

#CI
ci <- confint(olrrevised)
ci

#CI exp
exp(cbind(OR = coef(olrrevised), ci))

#overall effect size
ll.null = olrrev$null.deviance/-2
ll.proposed = olr$deviance/-2
(ll.null - ll.proposed) / ll.null
1 - pchisq(2*(ll.proposed - ll.null), df=(length(olr$coefficients)-1))

#mcfadden pseudo r2
require(pscl)
pR2(olrrevised)

#effects plot
 # only need to do once. 
library(effects)
plot(allEffects(olrrevised))

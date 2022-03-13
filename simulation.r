# On a particular production line
# The probability of a bulb being faulty is 0.08
# In a quality control test, items are selected at random from the production line
# It is assumed that the quality of a bulb is independent of the other
# The observation is performed 100 times
# it is known that that the control test is performed on exactly 100 bulbs at a time
# as the bulb production machine made 100 bulbs at a time
## R by default: X model the number of failure before first success
## Let X denote be "The number of bulbs picked up before the 1st faulty bulb is selected"
## Let success be "Faulty items"
## p = 0.08 & q = 0.92
set.seed(1)
n=100
geom_r=rgeom(n,0.08)
geom_r
range(geom_r)
#
# Histogram of the geom_r
#
x=hist(geom_r, breaks = "FD")
x
length(x$breaks)
hist(geom_r, breaks = "FD", ylim = c(0,35), xaxp=c(0,60,12),
     main = "Histogram of A Geometric r.v",
     xlab = "Number of Failure before the first faulty bulb",
     las = 1, col = "red")
x$counts
#
#Jitter function
#
geom_r_jitter=jitter(geom_r)
geom_r_jitter
range(geom_r_jitter)

134

#
# Histogram of the geom_r_jitter
#
hist(geom_r_jitter, breaks = "FD", ylim = c(0,35), xaxp = c(0,60,12),
     main = "Histogram of A Geometric r.v",
     sub = "Note: Plotted using jitter function",
     xlab = "Number of Failure before the first faulty bulb",
     las = 1, col = "blue", col.sub = 2)
#
# Comparing histogram of geom_r & geom_r_jitter
#
par(mfrow = c(1,2))
hist(geom_r, breaks = 50, ylim = c(0,10), xaxp=c(0,60,12),
     main = "Histogram of A Geometric r.v",
     xlab = "Number of Failure before the first faulty bulb",
     las = 1, col = "red")
hist(geom_r_jitter, breaks = 50, ylim = c(0,10), xaxp = c(0,60,12),
     main = "Histogram of A Geometric r.v",
     sub = "Note: Plotted using jitter function",
     xlab = "Number of Failure before the first faulty bulb",
     las = 1, col = "blue", col.sub = 2)
#
# The histogram & Density function of geom_r_jitter
#
par(mfrow = c(1,2))
hist(geom_r, breaks = "FD", ylim = c(0,35), xaxp=c(0,60,12),
     main = "Histogram of A Geometric r.v",
     xlab = "Number of Failure before the first faulty bulb",
     las = 1, col = "red")
hist(geom_r, freq = F, breaks = "FD", ylim = c(0,0.08), xaxp = c(0,60,12),
     main = "Density Function of the Geometric r.v",
     xlab = "Number of Failure before the first faulty bulb",
     las = 1, col = "yellow", col.sub = 2)
lines(density(geom_r), lwd = 2)
x
df.geom.x = data.frame(x$counts, x$density)
df.geom.x

135

# QUESTION 3(B)
#
#for geometric d routine
#
#
geom_d= dgeom(0:205,0.08)
geom_d
sum(geom_d)
length(geom_d)
#
# Histogram of geom_d
#
x1 = hist(geom_d, ylim = c(0,200), xaxp = c(0,0.08,8))
x1
hist(geom_d, ylim = c(0,200), xaxp = c(0,0.08,8),
     main = "Hist of Geo-Dist: X~Geo(0.08)",
     xlab = "probability",
     las = 1, col = "turquoise")
#
# x-y plot geom_d
#
# Let x be
x = seq(0,205, by = 1)
# Let y be
y = geom_d
plot(x,y, type = "h", yaxp = c(0,0.08,8), xaxp = c(0,205,5),
     main = "Geo-Dist: X~Geo(0.08)",
     xlab = "x", ylab = "P(X=x)", las = 1)
#
# Interpreting Histogram & x-y plot ofgeom_d
#
par(mfrow = c(1,2))
hist(geom_d, ylim = c(0,200), xaxp = c(0,0.08,8),
     main = "Hist of Geo-Dist: X~Geo(0.08)",
     xlab = "probability",
     las = 1, col = "turquoise")
plot(x,y, type = "h", yaxp = c(0,0.08,8), xaxp = c(0,205,5),
     main = "Geo-Dist: X~Geo(0.08)",
     xlab = "x", ylab = "P(X=x)", las = 1)
x1$counts

136

# QUESTION 3(C)
phi = function(u, prob){
  u = exp(1i*u)
  ch = prob/(1-((1-prob)*u))
}
func= function(u,x,prob){
  Re(phi(u,prob)*exp(-1i*u*x)/(2*pi*(1-exp(-1i*u))))
}
derive_integral = function(v,p1){
  integral = (0.5- integrate(func,-pi,0,x=v,prob=p1)$value-
                integrate(func,0,pi,x=v,prob=p1)$value)
  integral;
}
#
# Working Examples using derive_integral
#
#for x =0,1,2,3,4, to find pdf from given cdf
#use f(0)= a(1), f(1)= a(2)-a(1)
p4 = derive_integral(5,0.08) - derive_integral(4,0.08)
p4
dgeom(4,0.08)
p7 = derive_integral(8,0.08) - derive_integral(7,0.08)
p7
dgeom(7,0.08)
p_greater_10 = 1 - derive_integral(11, 0.08)
p_greater_10
sum(dgeom(11:205,0.08))
p_less_150 = derive_integral(150, 0.08)
p_less_150
sum(dgeom(0:149,0.08))
p_greater_equ_100 = 1 - derive_integral(100, 0.08)
p_greater_equ_100
sum(dgeom(100:205,0.08))
p_less_equ_97 = derive_integral(98, 0.08)
p_less_equ_97
sum(dgeom(0:97,0.08))

137

p_greater_80_less_120 = derive_integral(120,0.08) - derive_integral(81,0.08)
p_greater_80_less_120
sum(dgeom(81:119,0.08))
p_greater_equ_15_less_equ_30 = derive_integral(31,0.08) - derive_integral(15,0.08)
p_greater_equ_15_less_equ_30
sum(dgeom(15:30,0.08))
p_greater_75_less_equ_205 = derive_integral(206,0.08) - derive_integral(76,0.08)
p_greater_75_less_equ_205
sum(dgeom(76:205,0.08))
p_greater_equ_55_less_193 = derive_integral(193,0.08) - derive_integral(55,0.08)
p_greater_equ_55_less_193
sum(dgeom(55:192,0.08))
#
#loop for cdf
#
cdf_geometric = numeric(205)
for (i in 1:205){
  cdf_geometric[i] = derive_integral(v=i,0.08)
}
cdf_geometric
#
# Histogram of cdf_geometric (CDF)
#
cdf = hist(cdf_geometric, breaks = 10, ylim = c(0,200),xaxp = c(0,1,10),
           main = "Histogram of Cdf of the Geo-Dist",
           xlab = "x", ylab = expression(P("X"<="x")),
           las = 1, col = "green")
#
# Histogram of geom_d (PDF)
#
geom_d= dgeom(0:205,0.08)
geom_d
pdf = hist(geom_d, ylim = c(0,200), xaxp = c(0,0.08,8),
           main = "Histogram of Pdf of the Geo-Dist: X~Geo(0.08)",
           xlab = "probability", ylab = "P(X=x)",
           las = 1, col = "turquoise")

138

#
# Comparing Histogram of geom_d (PDF) & cdf_geometric (CDF)
#
par(mfrow = c(1,2))
pdf = hist(geom_d, ylim = c(0,200), xaxp = c(0,0.08,8),
           main = "Histogram of Pdf of the Geo-Dist: X~Geo(0.08)",
           xlab = "probability",
           las = 1, col = "turquoise")
cdf = hist(cdf_geometric, breaks = 10, ylim = c(0,200),xaxp = c(0,1,10),
           main = "Histogram of Cdf of the Geo-Dist",
           xlab = "x", ylab = expression(P("X"<="x")),
           las = 1, col = "green")
pdf$counts
cdf$counts
#
# x-y plot of cdf_geometric
#
x_cdf = seq(0,204,by=1)
y_cdf = cdf_geometric
plot(x_cdf, y_cdf, type = "s", xaxp = c(0,205,41), yaxp = c(0,1,10),
     main = "CDF of the Geo-Dist",
     xlab = "x", ylab = expression(P("X"<="x")), las = 1)
#
# x-y plot of geom_d (PDF)
#
x = seq(0,205, by = 1)
y = geom_d
plot(x,y, type = "h", yaxp = c(0,0.08,8), xaxp = c(0,205,5),
     main = "Pdf of the Geo-Dist: X~Geo(0.08)",
     xlab = "x", ylab = "P(X=x)", las = 1)
# Blue Example
p15 = derive_integral(16,0.08) - derive_integral(15,0.08)
p15
dgeom(15,0.08)
derive_integral(16,0.08)
derive_integral(15,0.08)

139

# Green Example
p30 = derive_integral(31,0.08) - derive_integral(30,0.08)
p30
dgeom(30,0.08)
p_less_equ_30 = derive_integral(31,0.08)
p_less_equ_30
p_less_equ_29 = derive_integral(30,0.08)
p_less_equ_29
# Red Example
p_less_equ_145 = derive_integral(146,0.08)
p_less_equ_145
sum(dgeom(0:145,0.08))

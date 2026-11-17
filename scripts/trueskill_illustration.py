# 同一個假設例子貫穿預測與更新；不取用真實玩家，也不作模型表現的證據。
import numpy as np
import polars as pl
import trueskill
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mu_a, sigma_a, mu_b, sigma_b, beta, epsilon = 0., 1., 1., .5, 1., .2
c = np.sqrt(sigma_a**2+sigma_b**2+2*beta**2)
mean = mu_a-mu_b
prob = [norm.cdf((-epsilon-mean)/c), norm.cdf((epsilon-mean)/c)-norm.cdf((-epsilon-mean)/c), norm.sf((epsilon-mean)/c)]
draw_probability = 2*norm.cdf(epsilon/(np.sqrt(2)*beta))-1
env = trueskill.TrueSkill(mu=0,sigma=1,beta=beta,tau=0,draw_probability=draw_probability,backend='scipy')
after_a, after_b = env.rate_1vs1(env.create_rating(mu_a,sigma_a),env.create_rating(mu_b,sigma_b))
# 直接積分本例的 posterior moments，核對圖中套件輸出的 Normal approximation。
for mu, sigma, opp_mu, opp_sigma, sign, after in [(mu_a,sigma_a,mu_b,sigma_b,1,after_a),(mu_b,sigma_b,mu_a,sigma_a,-1,after_b)]:
    density = lambda s: norm.pdf(s,mu,sigma)*norm.cdf((sign*(s-opp_mu)-epsilon)/np.sqrt(opp_sigma**2+2*beta**2))/prob[2]
    mass = quad(density,-np.inf,np.inf)[0]
    first = quad(lambda s:s*density(s),-np.inf,np.inf)[0]
    second = quad(lambda s:s*s*density(s),-np.inf,np.inf)[0]
    assert np.allclose([mass,first,np.sqrt(second-first*first)],[1.,after.mu,after.sigma],atol=1e-10)
rows = []
for name,mu,sigma,after in [('A',mu_a,sigma_a,after_a),('B',mu_b,sigma_b,after_b)]:
    rows.append(dict(player=name,prior_mu=mu,prior_sigma=sigma,posterior_mu=after.mu,posterior_sigma=after.sigma,
                     beta=beta,epsilon=epsilon,p_loss=prob[0],p_draw=prob[1],p_win=prob[2],difference_mean=mean,difference_sd=c,
                     outcome='A wins; hypothetical example'))
pl.DataFrame(rows).write_csv('outputs/tables/trueskill_illustration.csv')
plt.rcParams.update({'svg.fonttype':'path','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig, ax = plt.subplots(figsize=(11,3.5))
x = np.linspace(mean-4*c,mean+4*c,2000)
ax.plot(x,norm.pdf(x,mean,c),color='#33414a',lw=2)
for lo,hi,color,label,value in [(-np.inf,-epsilon,'#8498b0','A loses',prob[0]),(-epsilon,epsilon,'#d9b772','Draw',prob[1]),(epsilon,np.inf,'#4d9484','A wins',prob[2])]:
    mask = (x >= lo) & (x <= hi)
    xx = np.r_[max(lo,x[0]),x[mask],min(hi,x[-1])]
    ax.fill_between(xx,norm.pdf(xx,mean,c),color=color,alpha=.55,label=f'{label}: {100*value:.1f}%')
for v,label in [(-epsilon,r'$-\epsilon$'),(epsilon,r'$+\epsilon$')]:
    ax.axvline(v,color='#6a7075',ls=':',lw=1)
    ax.annotate(label,(v,0),xytext=(-14 if v<0 else 14,-22),textcoords='offset points',ha='center')
ax.set(xlabel=r'Performance difference $D=X_A-X_B$',ylabel='Density',xlim=(mean-3.5*c,mean+3.5*c),ylim=(0,None))
ax.xaxis.labelpad=16
ax.legend(frameon=False,loc='upper right')
ax.grid(axis='y',alpha=.15)
fig.tight_layout()
fig.savefig('outputs/figures/trueskill_illustration_probabilities.svg',bbox_inches='tight')
plt.close(fig)
fig, axes = plt.subplots(1,2,figsize=(11,3.6),sharex=True,sharey=True)
x = np.linspace(-3.5,3.5,800)
for ax,row in zip(axes,rows):
    ax.plot(x,norm.pdf(x,row['prior_mu'],row['prior_sigma']),color='#7f878d',ls='--',lw=2,label='Before: prior')
    ax.plot(x,norm.pdf(x,row['posterior_mu'],row['posterior_sigma']),color='#247a70',lw=2,label='After: Normal approximation')
    ax.set(title=f"Player {row['player']}",xlabel='Skill (arbitrary units)',ylabel='Density',ylim=(0,1.08))
    ax.text(.03,.94,f"Before: μ={row['prior_mu']:.2f}, σ={row['prior_sigma']:.2f}\nAfter:   μ={row['posterior_mu']:.2f}, σ={row['posterior_sigma']:.2f}",transform=ax.transAxes,va='top',fontsize=11)
    ax.grid(axis='y',alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
fig.tight_layout(rect=(0,0,1,.88))
fig.savefig('outputs/figures/trueskill_illustration_update.svg',bbox_inches='tight')
plt.close(fig)
print(pl.DataFrame(rows))

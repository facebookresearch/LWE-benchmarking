"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[936],{8546:(e,t,s)=>{s.r(t),s.d(t,{assets:()=>l,contentTitle:()=>c,default:()=>j,frontMatter:()=>d,metadata:()=>i,toc:()=>h});var n=s(4848),r=s(8453);const d={title:"Benchmark Results",description:"benchmark results page",hide_table_of_contents:!0},c="Benchmark Results",i={type:"mdx",permalink:"/benchmark",source:"@site/src/pages/benchmark.md",title:"Benchmark Results",description:"benchmark results page",frontMatter:{title:"Benchmark Results",description:"benchmark results page",hide_table_of_contents:!0},unlisted:!1},l={},h=[];function x(e){const t={h1:"h1",p:"p",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",...(0,r.R)(),...e.components};return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(t.h1,{id:"benchmark-results",children:"Benchmark Results"}),"\n",(0,n.jsxs)(t.p,{children:[(0,n.jsx)(t.strong,{children:"Performance of all attacks on benchmark settings."})," Best Hamming weight (h) for secret recovered per setting/attack, time in hours needed to recover this secret, and machines used. Highest h per setting is ",(0,n.jsx)(t.strong,{children:"bold"}),'. All Kyber secrets are binomial, and HE secrets are ternary. First three attacks (uSVP, SALSA, CC) solve Search-LWE; MITM* solves Decision LWE. "Total hrs" is total attack time assuming full parallelization.']}),"\n",(0,n.jsxs)(t.table,{children:[(0,n.jsx)(t.thead,{children:(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.th,{children:(0,n.jsx)(t.strong,{children:"Attack"})}),(0,n.jsx)(t.th,{children:(0,n.jsx)(t.strong,{children:"Results"})}),(0,n.jsxs)(t.th,{children:[(0,n.jsx)(t.strong,{children:"n=256, k=2, logq=12"})," binomial"]}),(0,n.jsxs)(t.th,{children:[(0,n.jsx)(t.strong,{children:"n=256, k=2, logq=28"})," binomial"]}),(0,n.jsxs)(t.th,{children:[(0,n.jsx)(t.strong,{children:"n=256, k=3, logq=35"})," binomial"]}),(0,n.jsxs)(t.th,{children:[(0,n.jsx)(t.strong,{children:"n=1024, logq=26"})," ternary"]}),(0,n.jsxs)(t.th,{children:[(0,n.jsx)(t.strong,{children:"n=1024, logq=29"})," ternary"]}),(0,n.jsxs)(t.th,{children:[(0,n.jsx)(t.strong,{children:"n=1024, logq=50"})," ternary"]})]})}),(0,n.jsxs)(t.tbody,{children:[(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"uSVP"})}),(0,n.jsx)(t.td,{children:"Best h"}),(0,n.jsx)(t.td,{children:"-"}),(0,n.jsx)(t.td,{children:"-"}),(0,n.jsx)(t.td,{children:"-"}),(0,n.jsx)(t.td,{children:"-"}),(0,n.jsx)(t.td,{children:"-"}),(0,n.jsx)(t.td,{children:"-"})]}),(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{}),(0,n.jsx)(t.td,{children:"Recover hrs (1 CPU)"}),(0,n.jsx)(t.td,{children:">1100"}),(0,n.jsx)(t.td,{children:">1100"}),(0,n.jsx)(t.td,{children:">1100"}),(0,n.jsx)(t.td,{children:">1300"}),(0,n.jsx)(t.td,{children:">1300"}),(0,n.jsx)(t.td,{children:">1300"})]}),(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"SALSA"})}),(0,n.jsx)(t.td,{children:"Best h"}),(0,n.jsx)(t.td,{children:"9"}),(0,n.jsx)(t.td,{children:"18"}),(0,n.jsx)(t.td,{children:"16"}),(0,n.jsx)(t.td,{children:"8"}),(0,n.jsx)(t.td,{children:"10"}),(0,n.jsx)(t.td,{children:"17"})]}),(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{}),(0,n.jsx)(t.td,{children:"Total hrs"}),(0,n.jsx)(t.td,{children:"36"}),(0,n.jsx)(t.td,{children:"27"}),(0,n.jsx)(t.td,{children:"39"}),(0,n.jsx)(t.td,{children:"34.9"}),(0,n.jsx)(t.td,{children:"49.4"}),(0,n.jsx)(t.td,{children:"29.1"})]}),(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"CC"})}),(0,n.jsx)(t.td,{children:"Best h"}),(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"11"})}),(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"25"})}),(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"19"})}),(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"12"})}),(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"12"})}),(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"20"})})]}),(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{}),(0,n.jsx)(t.td,{children:"Total hrs"}),(0,n.jsx)(t.td,{children:"28.1"}),(0,n.jsx)(t.td,{children:"53"}),(0,n.jsx)(t.td,{children:"34"}),(0,n.jsx)(t.td,{children:"21.5"}),(0,n.jsx)(t.td,{children:"31.7"}),(0,n.jsx)(t.td,{children:"28"})]}),(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{children:(0,n.jsx)(t.strong,{children:"MiTM (Decision LWE)"})}),(0,n.jsx)(t.td,{children:"Best h"}),(0,n.jsx)(t.td,{children:"4"}),(0,n.jsx)(t.td,{children:"12"}),(0,n.jsx)(t.td,{children:"14"}),(0,n.jsx)(t.td,{children:"9"}),(0,n.jsx)(t.td,{children:"9"}),(0,n.jsx)(t.td,{children:"16"})]}),(0,n.jsxs)(t.tr,{children:[(0,n.jsx)(t.td,{}),(0,n.jsx)(t.td,{children:"Total hrs"}),(0,n.jsx)(t.td,{children:"0.7"}),(0,n.jsx)(t.td,{children:"1.61"}),(0,n.jsx)(t.td,{children:"29.4"}),(0,n.jsx)(t.td,{children:"65"}),(0,n.jsx)(t.td,{children:"13"}),(0,n.jsx)(t.td,{children:"15.5"})]})]})]})]})}function j(e={}){const{wrapper:t}={...(0,r.R)(),...e.components};return t?(0,n.jsx)(t,{...e,children:(0,n.jsx)(x,{...e})}):x(e)}},8453:(e,t,s)=>{s.d(t,{R:()=>c,x:()=>i});var n=s(6540);const r={},d=n.createContext(r);function c(e){const t=n.useContext(d);return n.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function i(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:c(e.components),n.createElement(d.Provider,{value:t},e.children)}}}]);
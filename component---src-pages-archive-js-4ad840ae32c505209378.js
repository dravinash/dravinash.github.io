"use strict";(self.webpackChunkv1=self.webpackChunkv1||[]).push([[231],{3229:function(e,t,l){l.r(t);var a=l(6540),n=l(8154),r=l(7383),i=l(8344),o=l(418),c=l(6462),s=l(4612),m=l(1626);const{colors:d,fonts:h,fontSizes:p}=m.w4,f=(0,s.default)(m.gZ).withConfig({displayName:"archive__StyledMainContainer",componentId:"sc-m3t4e8-0"})([""]),u=s.default.div.withConfig({displayName:"archive__StyledTableContainer",componentId:"sc-m3t4e8-1"})(["margin:100px -20px;",";"],m.$_.tablet`
    margin: 100px -10px;
  `),E=s.default.table.withConfig({displayName:"archive__StyledTable",componentId:"sc-m3t4e8-2"})(["width:100%;border-collapse:collapse;.hide-on-mobile{",";}tbody tr{transition:",";&:hover,&:focus{background-color:",";}}th,td{cursor:default;line-height:1.5;padding:10px 20px;",";}th{text-align:left;}td{&.year{width:10%;",";}&.title{padding-top:15px;color:",";font-size:",";font-weight:700;}&.company{width:15%;padding-top:15px;font-size:",";}&.tech{font-size:",";font-family:",";.separator{margin:0 5px;}span{display:inline-block;}}&.links{span{display:flex;align-items:center;a{",";}a + a{margin-left:10px;}svg{width:20px;height:20px;}}}}"],m.$_.tablet`
      display: none;
    `,m.w4.transition,d.lightNavy,m.$_.tablet`
      padding: 10px;
    `,m.$_.tablet`
        font-size: ${p.sm};
      `,d.lightestSlate,p.xl,p.lg,p.xs,h.SFMono,m.vE.flexCenter);t.default=e=>{let{location:t,data:l}=e;const s=l.allMarkdownRemark.edges,m=(0,a.useRef)(null),d=(0,a.useRef)(null),h=(0,a.useRef)([]);return(0,a.useEffect)((()=>{r.A.reveal(m.current,(0,i.srConfig)()),r.A.reveal(d.current,(0,i.srConfig)()),h.current.forEach(((e,t)=>r.A.reveal(e,(0,i.srConfig)(10*t))))}),[]),a.createElement(o.PE,{location:t},a.createElement(n.m,null,a.createElement("title",null,"Archive | Dr. Avinash Kumar Singh"),a.createElement("link",{rel:"canonical",href:"https://chandrikadeb7.github.io/archive"})),a.createElement(f,null,a.createElement("header",{ref:m},a.createElement("h1",{className:"big-title"},"Archive"),a.createElement("p",{className:"subtitle"},"A big list of things I’ve worked on")),a.createElement(u,{ref:d},a.createElement(E,null,a.createElement("thead",null,a.createElement("tr",null,a.createElement("th",null,"Year"),a.createElement("th",null,"Title"),a.createElement("th",{className:"hide-on-mobile"},"Made at"),a.createElement("th",{className:"hide-on-mobile"},"Built with"),a.createElement("th",null,"Link"))),a.createElement("tbody",null,s.length>0&&s.map(((e,t)=>{let{node:l}=e;const{date:n,github:r,external:i,title:o,tech:s,company:m}=l.frontmatter;return a.createElement("tr",{key:t,ref:e=>h.current[t]=e},a.createElement("td",{className:"overline year"},`${new Date(n).getFullYear()}`),a.createElement("td",{className:"title"},o),a.createElement("td",{className:"company hide-on-mobile"},m?a.createElement("span",null,m):a.createElement("span",null,"—")),a.createElement("td",{className:"tech hide-on-mobile"},s.length>0&&s.map(((e,t)=>a.createElement("span",{key:t},e,"",t!==s.length-1&&a.createElement("span",{className:"separator"},"·"))))),a.createElement("td",{className:"links"},a.createElement("span",null,i&&a.createElement("a",{href:i,target:"_blank",rel:"nofollow noopener noreferrer","aria-label":"External Link"},a.createElement(c.i1,{name:"External"})),r&&a.createElement("a",{href:r,target:"_blank",rel:"nofollow noopener noreferrer","aria-label":"GitHub Link"},a.createElement(c.i1,{name:"GitHub"})))))})))))))}}}]);
//# sourceMappingURL=component---src-pages-archive-js-4ad840ae32c505209378.js.map
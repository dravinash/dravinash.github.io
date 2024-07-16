(self.webpackChunkv1=self.webpackChunkv1||[]).push([[668],{5332:function(e,t,n){"use strict";n.r(t);var o=n(6540),r=n(8007),a=n(8154),i=n(249),l=n.n(i),u=n(418),f=n(6462),s=n(4612),c=n(1626);const{colors:d,fontSizes:p,fonts:g}=c.w4,x=(0,s.default)(c.gZ).withConfig({displayName:"blog_posts__StyledMainContainer",componentId:"sc-f998de-0"})(["& > header{text-align:center;margin-bottom:100px;}footer{",";margin-top:20px;width:100%;}"],c.vE.flexBetween),m=s.default.div.withConfig({displayName:"blog_posts__StyledGrid",componentId:"sc-f998de-1"})(["margin-top:50px;.posts{display:grid;text-align:justify;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));grid-gap:15px;position:relative;",";}"],c.$_.desktop`grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));`),h=s.default.div.withConfig({displayName:"blog_posts__StyledPostInner",componentId:"sc-f998de-2"})(["",";",";flex-direction:column;align-items:flex-start;text-align:justify;position:relative;padding:2rem 1.75rem;height:100%;border-radius:",";transition:",";background-color:",";header,a{width:100%;}"],c.vE.boxShadow,c.vE.flexBetween,c.w4.borderRadius,c.w4.transition,d.lightNavy),v=s.default.div.withConfig({displayName:"blog_posts__StyledPost",componentId:"sc-f998de-3"})(["transition:",";cursor:default;text-align:justify;&:hover,&:focus{outline:0;","{transform:translateY(-5px);}}"],c.w4.transition,h),y=s.default.div.withConfig({displayName:"blog_posts__StyledPostHeader",componentId:"sc-f998de-4"})(["",";margin-bottom:30px;"],c.vE.flexBetween),b=s.default.div.withConfig({displayName:"blog_posts__StyledFolder",componentId:"sc-f998de-5"})(["color:",";svg{width:0px;height:0px;}"],d.green),E=s.default.h5.withConfig({displayName:"blog_posts__StyledPostName",componentId:"sc-f998de-6"})(["margin:0 0 10px;font-size:",";color:",";"],p.xxl,d.lightestSlate),_=s.default.div.withConfig({displayName:"blog_posts__StyledPostDescription",componentId:"sc-f998de-7"})(["font-size:17px;text-align:justify;color:",";"],d.lightSlate),S=s.default.span.withConfig({displayName:"blog_posts__StyledDate",componentId:"sc-f998de-8"})(["text-transform:uppercase;text-align:justify;font-family:",";font-size:",";color:",";"],g.SFMono,p.xs,d.lightSlate),w=s.default.ul.withConfig({displayName:"blog_posts__StyledTags",componentId:"sc-f998de-9"})(["display:flex;text-align:justify;align-items:flex-end;flex-wrap:wrap;padding:0;margin:0;list-style:none;li{font-family:",";font-size:",";color:",";line-height:1.75;text-align:justify;margin-right:15px;&:last-of-type{margin-right:0;}a{",";}}"],g.SFMono,p.xs,d.green,c.vE.inlineLink);t.default=e=>{let{location:t,data:n}=e;const i=n.allMarkdownRemark.edges;return o.createElement(u.PE,{location:t},o.createElement(a.m,null,o.createElement("title",null,"Blog | Dr. Avinash Kumar Singh"),o.createElement("link",{rel:"canonical",href:"https://github.com/dravinash"})),o.createElement(x,null,o.createElement("header",null,o.createElement("h1",{className:"big-title"},"Blog Posts"),o.createElement("p",{className:"subtitle"},o.createElement("a",{href:"http://avinashkumarsingh.in/blog_posts",target:"_blank",rel:"noopener noreferrer"},"a collection of articles"))),o.createElement(m,null,o.createElement("div",{className:"posts"},i.length>0&&i.map(((e,t)=>{let{node:n}=e;const{frontmatter:a}=n,{title:i,description:u,slug:s,date:c,tags:d}=a,p=new Date(c);return o.createElement(v,{key:t,tabIndex:"0"},o.createElement(h,null,o.createElement("header",null,o.createElement(r.Link,{to:s},o.createElement(y,null,o.createElement(b,null,o.createElement(f.hf,null))),o.createElement(E,null,i),o.createElement(_,null,u))),o.createElement("footer",null,o.createElement(S,null,`${p.toLocaleDateString()}`),o.createElement(w,null,d.map(((e,t)=>o.createElement("li",{key:t},o.createElement(r.Link,{to:`/blog_posts/tags/${l()(e)}/`},"#",e))))))))}))))))}},1873:function(e,t,n){var o=n(9325).Symbol;e.exports=o},4932:function(e){e.exports=function(e,t){for(var n=-1,o=null==e?0:e.length,r=Array(o);++n<o;)r[n]=t(e[n],n,e);return r}},882:function(e){e.exports=function(e,t,n,o){var r=-1,a=null==e?0:e.length;for(o&&a&&(n=e[++r]);++r<a;)n=t(n,e[r],r,e);return n}},1733:function(e){var t=/[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g;e.exports=function(e){return e.match(t)||[]}},2552:function(e,t,n){var o=n(1873),r=n(659),a=n(9350),i=o?o.toStringTag:void 0;e.exports=function(e){return null==e?void 0===e?"[object Undefined]":"[object Null]":i&&i in Object(e)?r(e):a(e)}},4552:function(e){e.exports=function(e){return function(t){return null==e?void 0:e[t]}}},7556:function(e,t,n){var o=n(1873),r=n(4932),a=n(6449),i=n(4394),l=o?o.prototype:void 0,u=l?l.toString:void 0;e.exports=function e(t){if("string"==typeof t)return t;if(a(t))return r(t,e)+"";if(i(t))return u?u.call(t):"";var n=t+"";return"0"==n&&1/t==-1/0?"-0":n}},5539:function(e,t,n){var o=n(882),r=n(828),a=n(6645),i=RegExp("['’]","g");e.exports=function(e){return function(t){return o(a(r(t).replace(i,"")),e,"")}}},4647:function(e,t,n){var o=n(4552)({"À":"A","Á":"A","Â":"A","Ã":"A","Ä":"A","Å":"A","à":"a","á":"a","â":"a","ã":"a","ä":"a","å":"a","Ç":"C","ç":"c","Ð":"D","ð":"d","È":"E","É":"E","Ê":"E","Ë":"E","è":"e","é":"e","ê":"e","ë":"e","Ì":"I","Í":"I","Î":"I","Ï":"I","ì":"i","í":"i","î":"i","ï":"i","Ñ":"N","ñ":"n","Ò":"O","Ó":"O","Ô":"O","Õ":"O","Ö":"O","Ø":"O","ò":"o","ó":"o","ô":"o","õ":"o","ö":"o","ø":"o","Ù":"U","Ú":"U","Û":"U","Ü":"U","ù":"u","ú":"u","û":"u","ü":"u","Ý":"Y","ý":"y","ÿ":"y","Æ":"Ae","æ":"ae","Þ":"Th","þ":"th","ß":"ss","Ā":"A","Ă":"A","Ą":"A","ā":"a","ă":"a","ą":"a","Ć":"C","Ĉ":"C","Ċ":"C","Č":"C","ć":"c","ĉ":"c","ċ":"c","č":"c","Ď":"D","Đ":"D","ď":"d","đ":"d","Ē":"E","Ĕ":"E","Ė":"E","Ę":"E","Ě":"E","ē":"e","ĕ":"e","ė":"e","ę":"e","ě":"e","Ĝ":"G","Ğ":"G","Ġ":"G","Ģ":"G","ĝ":"g","ğ":"g","ġ":"g","ģ":"g","Ĥ":"H","Ħ":"H","ĥ":"h","ħ":"h","Ĩ":"I","Ī":"I","Ĭ":"I","Į":"I","İ":"I","ĩ":"i","ī":"i","ĭ":"i","į":"i","ı":"i","Ĵ":"J","ĵ":"j","Ķ":"K","ķ":"k","ĸ":"k","Ĺ":"L","Ļ":"L","Ľ":"L","Ŀ":"L","Ł":"L","ĺ":"l","ļ":"l","ľ":"l","ŀ":"l","ł":"l","Ń":"N","Ņ":"N","Ň":"N","Ŋ":"N","ń":"n","ņ":"n","ň":"n","ŋ":"n","Ō":"O","Ŏ":"O","Ő":"O","ō":"o","ŏ":"o","ő":"o","Ŕ":"R","Ŗ":"R","Ř":"R","ŕ":"r","ŗ":"r","ř":"r","Ś":"S","Ŝ":"S","Ş":"S","Š":"S","ś":"s","ŝ":"s","ş":"s","š":"s","Ţ":"T","Ť":"T","Ŧ":"T","ţ":"t","ť":"t","ŧ":"t","Ũ":"U","Ū":"U","Ŭ":"U","Ů":"U","Ű":"U","Ų":"U","ũ":"u","ū":"u","ŭ":"u","ů":"u","ű":"u","ų":"u","Ŵ":"W","ŵ":"w","Ŷ":"Y","ŷ":"y","Ÿ":"Y","Ź":"Z","Ż":"Z","Ž":"Z","ź":"z","ż":"z","ž":"z","Ĳ":"IJ","ĳ":"ij","Œ":"Oe","œ":"oe","ŉ":"'n","ſ":"s"});e.exports=o},4840:function(e,t,n){var o="object"==typeof n.g&&n.g&&n.g.Object===Object&&n.g;e.exports=o},659:function(e,t,n){var o=n(1873),r=Object.prototype,a=r.hasOwnProperty,i=r.toString,l=o?o.toStringTag:void 0;e.exports=function(e){var t=a.call(e,l),n=e[l];try{e[l]=void 0;var o=!0}catch(u){}var r=i.call(e);return o&&(t?e[l]=n:delete e[l]),r}},5434:function(e){var t=/[a-z][A-Z]|[A-Z]{2}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/;e.exports=function(e){return t.test(e)}},9350:function(e){var t=Object.prototype.toString;e.exports=function(e){return t.call(e)}},9325:function(e,t,n){var o=n(4840),r="object"==typeof self&&self&&self.Object===Object&&self,a=o||r||Function("return this")();e.exports=a},2225:function(e){var t="\\ud800-\\udfff",n="\\u2700-\\u27bf",o="a-z\\xdf-\\xf6\\xf8-\\xff",r="A-Z\\xc0-\\xd6\\xd8-\\xde",a="\\xac\\xb1\\xd7\\xf7\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf\\u2000-\\u206f \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000",i="["+a+"]",l="\\d+",u="["+n+"]",f="["+o+"]",s="[^"+t+a+l+n+o+r+"]",c="(?:\\ud83c[\\udde6-\\uddff]){2}",d="[\\ud800-\\udbff][\\udc00-\\udfff]",p="["+r+"]",g="(?:"+f+"|"+s+")",x="(?:"+p+"|"+s+")",m="(?:['’](?:d|ll|m|re|s|t|ve))?",h="(?:['’](?:D|LL|M|RE|S|T|VE))?",v="(?:[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]|\\ud83c[\\udffb-\\udfff])?",y="[\\ufe0e\\ufe0f]?",b=y+v+("(?:\\u200d(?:"+["[^"+t+"]",c,d].join("|")+")"+y+v+")*"),E="(?:"+[u,c,d].join("|")+")"+b,_=RegExp([p+"?"+f+"+"+m+"(?="+[i,p,"$"].join("|")+")",x+"+"+h+"(?="+[i,p+g,"$"].join("|")+")",p+"?"+g+"+"+m,p+"+"+h,"\\d*(?:1ST|2ND|3RD|(?![123])\\dTH)(?=\\b|[a-z_])","\\d*(?:1st|2nd|3rd|(?![123])\\dth)(?=\\b|[A-Z_])",l,E].join("|"),"g");e.exports=function(e){return e.match(_)||[]}},828:function(e,t,n){var o=n(4647),r=n(3222),a=/[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g,i=RegExp("[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]","g");e.exports=function(e){return(e=r(e))&&e.replace(a,o).replace(i,"")}},6449:function(e){var t=Array.isArray;e.exports=t},346:function(e){e.exports=function(e){return null!=e&&"object"==typeof e}},4394:function(e,t,n){var o=n(2552),r=n(346);e.exports=function(e){return"symbol"==typeof e||r(e)&&"[object Symbol]"==o(e)}},249:function(e,t,n){var o=n(5539)((function(e,t,n){return e+(n?"-":"")+t.toLowerCase()}));e.exports=o},3222:function(e,t,n){var o=n(7556);e.exports=function(e){return null==e?"":o(e)}},6645:function(e,t,n){var o=n(1733),r=n(5434),a=n(3222),i=n(2225);e.exports=function(e,t,n){return e=a(e),void 0===(t=n?void 0:t)?r(e)?i(e):o(e):e.match(t)||[]}}}]);
//# sourceMappingURL=component---src-pages-blog-posts-index-js-dbd7cd903c074165ae62.js.map
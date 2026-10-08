// v5 ordered action interpreter. One lane owns one independent batch sample.
// All state evolution, loss seeds and reverse mode execute on device.
int atlas_dynamic_int(float x){return as_type<int>(x);}
float atlas_dynamic_bits(int x){return as_type<float>(x);}
bool atlas_dynamic_cache_enabled(device const ulong* m){return (m[13]==34&&m[33]==1)||(m[13]==36&&m[33]==2);}
void atlas_dynamic_cache_initialize(device const ulong* m,device const float* p,device float* tape,ulong b){
 if(m[13]!=36||m[33]!=2)return;
 ulong base=b*m[1]*m[8],end=base+m[1]*m[8];
 for(ulong j=0;j<m[35];j++){
  ulong q=m[34]+3*j,at=ulong(uint(atlas_dynamic_int(p[q])));
  if(at>=base&&at<end){tape[at]=1;tape[at+1]=p[q+1];tape[at+2]=p[q+2];tape[at+3]=0;}
 }
}
ulong atlas_dynamic_cache_slot(device const float* p,ulong noise,ulong stream){return ulong(uint(atlas_dynamic_int(p[noise+97+stream])));}
bool atlas_dynamic_cache_origin(device const float* p,device const float* cache,ulong noise,ulong stream){
 ulong at=atlas_dynamic_cache_slot(p,noise,stream);return cache[at]==1&&uint(atlas_dynamic_int(cache[at+3]))==uint(noise+1);
}
// Sparse, canonical leaf accumulation for a unit rate VJP. No host derivative
// or device-global gradient scratch is used by the probe.
bool atlas_dynamic_probe_add(ulong key,float value,thread ulong* keys,thread float* leaves,thread ulong& used){
 ulong k=0;while(k<used&&keys[k]!=key)k++;
 if(k==used){if(used>=128)return false;keys[k]=key;leaves[k]=0;used++;}
 leaves[k]+=value;return isfinite(leaves[k]);
}
bool atlas_dynamic_equation_vjp(device const ulong* m,device const float* p,
 ulong start,ulong constants,ulong count,ulong parameter_index,
 thread const float* value,thread const bool* seen,thread float* adjoint,
 device float* gradient,ulong gradient_base,thread float* state_gradient,
 thread const int* sources,ulong context_count,bool probe,thread bool& has_vjp,uint activity){
 bool active[128];ulong keys[128],used=0;float leaves[128];has_vjp=false;
 ulong mask_base=activity==2?m[13]-m[5]:m[20];
 if(probe||activity)for(ulong j=0;j<count;j++){
  active[j]=false;if(!seen[j])continue;
  ulong q=start+4*j,op=m[q],a=m[q+1],b=m[q+2],slot=m[q+3];
  switch(op){
   case 0:case 15:{ulong index=op==0?0:a;
    if(activity==2){active[j]=true;break;}
    if(index>=context_count||sources[index]<0||ulong(sources[index])>=m[4])return false;
    active[j]=!m[m[16]+ulong(sources[index])];break;}
   case 2:active[j]=p[mask_base+slot]!=0;break;
   case 17:active[j]=p[mask_base+slot+parameter_index]!=0;break;
   case 24:active[j]=p[mask_base+slot+m[a+parameter_index]]!=0;break;
   case 48:{int index=atlas_dynamic_int(value[a]);if(index<0||ulong(index)>=b)return false;
    active[j]=p[mask_base+slot+ulong(index)]!=0;break;}
   case 25:{float c=p[constants+j];
    if(value[b]!=floor(value[b])||value[b]<0||value[b]>=float(m[slot+2]))return false;
    ulong row=ulong(min(max((value[a]/c+0.5f)/float(m[slot+3]),0.0f),float(m[slot+1]-1)));
    active[j]=p[mask_base+m[slot]+row*m[slot+2]+ulong(value[b])]!=0;break;}
   case 3:case 4:case 5:case 6:case 47:active[j]=active[a]||active[b];break;
   case 7:case 8:case 9:case 10:case 11:case 12:case 13:case 14:case 30:case 52:active[j]=active[a];break;
   case 20:active[j]=active[value[a]<=value[b]?a:b];break;
   case 21:active[j]=active[value[a]>=value[b]?a:b];break;
   case 22:active[j]=active[value[a]==1?b:slot];break;
   case 54:active[j]=active[b];break;
   case 56:case 57:active[j]=active[a]||active[b];break;
   case 26:case 27:active[j]=active[a];break;
   case 28:active[j]=active[a]||(value[a]!=0&&active[b]);break;
   case 29:active[j]=active[a]||(value[a]!=1&&active[b]);break;
  }
 }
 for(ulong j=count;j-->0;){
  if((probe||activity)&&!active[j]){adjoint[j]=0;continue;}
  float g=adjoint[j];if(!isfinite(g))return false;if(g==0||!seen[j])continue;
  ulong q=start+4*j,op=m[q],a=m[q+1],b=m[q+2],slot=m[q+3];float c=p[constants+j];
  switch(op){
   case 0:case 15:{ulong index=op==0?0:a;
    if(probe){if(!atlas_dynamic_probe_add(m[5]+ulong(sources[index]),g,keys,leaves,used))return false;}
    else state_gradient[index]+=g;break;}
   case 2:case 17:{ulong index=slot+(op==17?parameter_index:0);
    if(probe){if(!atlas_dynamic_probe_add(index,g,keys,leaves,used))return false;}
    else gradient[gradient_base+index]+=g;break;}
   case 24:{ulong index=slot+m[a+parameter_index];
    if(probe){if(!atlas_dynamic_probe_add(index,g,keys,leaves,used))return false;}
    else gradient[gradient_base+index]+=g;break;}
   case 48:{int index=atlas_dynamic_int(value[a]);if(index<0||ulong(index)>=b)return false;ulong cell=slot+ulong(index);
    if(probe){if(!atlas_dynamic_probe_add(cell,g,keys,leaves,used))return false;}
    else gradient[gradient_base+cell]+=g;break;}
   case 25:{if(value[b]!=floor(value[b])||value[b]<0||value[b]>=float(m[slot+2]))return false;
    ulong row=ulong(min(max((value[a]/c+0.5f)/float(m[slot+3]),0.0f),float(m[slot+1]-1)));
    ulong cell=m[slot]+row*m[slot+2]+ulong(value[b]);
    if(probe){if(!atlas_dynamic_probe_add(cell,g,keys,leaves,used))return false;}
    else gradient[gradient_base+cell]+=g;break;}
   case 3:adjoint[a]+=g;adjoint[b]+=g;break;case 4:adjoint[a]+=g;adjoint[b]-=g;break;
   case 5:adjoint[a]+=g*value[b];adjoint[b]+=g*value[a];break;
   case 6:adjoint[a]+=g/value[b];adjoint[b]-=g*value[a]/(value[b]*value[b]);break;
   case 7:adjoint[a]-=g;break;case 8:adjoint[a]+=g*value[j];break;
   case 9:adjoint[a]+=g/value[a];break;case 10:adjoint[a]+=g*(1-value[j]*value[j]);break;
   case 11:adjoint[a]+=g/(2*value[j]);break;case 12:adjoint[a]+=g*cos(value[a]);break;
   case 13:adjoint[a]-=g*sin(value[a]);break;
   case 52:adjoint[a]+=g*atlas_dynamic_math_grad(slot,value[a]);break;
   case 14:if(c!=0){float z=pow(abs(value[a]),c-1);if(value[a]<0&&fmod(abs(c-1),2.0f)==1)z=-z;adjoint[a]+=g*c*z;}break;
   case 20:adjoint[value[a]<=value[b]?a:b]+=g;break;case 21:adjoint[value[a]>=value[b]?a:b]+=g;break;
   case 22:adjoint[value[a]==1?b:slot]+=g;break;
   case 54:adjoint[b]+=g;break;
   case 56:adjoint[a]+=g*value[b];adjoint[b]+=g*value[a];break;
   case 57:adjoint[a]+=g*(1-value[b]);adjoint[b]+=g*(1-value[a]);break;
   case 26:case 27:{float z=1+value[b]*abs(value[a]);adjoint[a]+=g*value[slot]/(z*z);break;}
   case 28:if(value[a]==0)adjoint[a]+=g;else{adjoint[a]+=g*value[b];adjoint[b]+=g;}break;
   case 29:if(value[a]==1)adjoint[a]+=g;else{adjoint[a]+=g*(1-value[b]);adjoint[b]+=g;}break;
   case 30:adjoint[a]-=g;break;
   case 47:adjoint[a]+=g;adjoint[b]-=g*round((value[a]-value[j])/value[b]);break;
  }
 }
 if(probe)for(ulong k=0;k<used;k++)has_vjp=has_vjp||leaves[k]!=0;
 return true;
}

float atlas_dynamic_equation_scored(device const ulong* m,device const float* p,device const float* weights,
 ulong header,ulong parameter_index,float time,ulong exact_time,ulong clocks,ulong noise,
 thread const float* state,device float* gradient,ulong gradient_base,
 float seed,bool backward,thread float* state_gradient,thread bool& valid,
 float score_loss,bool score_enabled,thread uint& scored,thread uint& needed_mask,thread const int* sources,ulong context_count,uint activity,device float* cache,bool cache_store) {
 valid=false;
 bool probe_only=activity==2?(m[80]&&m[m[15]+4]!=0):m[9]==2;
 ulong control=activity==2?m[15]:m[14];
 ulong count=m[header]&255,start=m[header+1],constants=m[header+2];
 float value[128],adjoint[128];bool seen[128];ulong stack[128];
 bool lazy=false;for(ulong j=0;j<count;j++){seen[j]=false;adjoint[j]=0;ulong op=m[start+4*j];lazy=lazy||op==53||op==22||op==28||op==29||op==42;}
 ulong depth=1,cursor=0;stack[0]=count-1;
 while(lazy?depth>0:cursor<count){
  ulong j=lazy?stack[depth-1]:cursor,q=start+4*j,op=m[q],a=m[q+1],b=m[q+2],slot=m[q+3];
  if(lazy){
   if(seen[j]){depth--;continue;}
   ulong dep[3];ulong length=0;
   if(op==22||op==42){
    if(!seen[a]){if(depth>=128)return NAN;stack[depth++]=a;continue;}
    if(value[a]!=0&&value[a]!=1)return NAN;dep[0]=value[a]==1?b:slot;length=1;
   }else if(op==28||op==29){
    if(!seen[a]){if(depth>=128)return NAN;stack[depth++]=a;continue;}
    if(value[a]!=0&&value[a]!=1)return NAN;
    dep[0]=b;length=((op==28&&value[a]==0)||(op==29&&value[a]==1))?0:1;
   }else if(op==51){dep[0]=a;dep[1]=b;dep[2]=m[slot];length=m[slot+2]?2:3;}
   else if(op==26||op==27){dep[0]=a;dep[1]=b;dep[2]=slot;length=3;}
   else if((op>=3&&op<=6)||op==20||op==21||op==25||op==31||op==32||op==40||op==41||op==43||op==46||op==47||op==54||op==55||op==56||op==57){dep[0]=a;dep[1]=b;length=2;}
   else if(op==53){
    bool rate_needed=true;
    if(m[header]&1024){ulong at=atlas_dynamic_cache_slot(p,noise,b);
     rate_needed=cache[at]!=1||(backward&&atlas_dynamic_cache_origin(p,cache,noise,b));}
    dep[0]=a;length=rate_needed?1:0;
   }else if((op>=7&&op<=14)||op==30||op==38||op==39||op==44||op==45||op==48||op==49||op==52){dep[0]=a;length=1;}
   bool waiting=false;for(ulong k=0;k<length;k++)if(!seen[dep[k]]){
    if(depth>=128)return NAN;stack[depth++]=dep[k];waiting=true;break;
   }
   if(waiting)continue;
  }
  float c=p[constants+j],z=0;
  switch(op){
   case 0:z=state[0];break;case 15:z=state[a];break;case 1:z=c;break;
   case 2:z=weights[slot];break;case 17:z=weights[slot+parameter_index];break;
   case 24:z=weights[slot+m[a+parameter_index]];break;
   case 48:case 49:{int index=atlas_dynamic_int(value[a]);if(index<0||ulong(index)>=b)return NAN;z=weights[slot+ulong(index)];break;}
   case 25:{if(value[b]!=floor(value[b])||value[b]<0||value[b]>=float(m[slot+2]))return NAN;
    ulong row=ulong(min(max((value[a]/c+0.5f)/float(m[slot+3]),0.0f),float(m[slot+1]-1)));
    ulong cell=m[slot]+row*m[slot+2]+ulong(value[b]);z=weights[cell];break;}
   case 50:{ulong bits=b?atlas_clock_from_float(time):exact_time;z=atlas_dynamic_bits(int(uint(bits>>(32*a))));break;}
   case 51:{ulong last=ulong(uint(atlas_dynamic_int(value[a])))|(ulong(uint(atlas_dynamic_int(value[b])))<<32);
    if(((last>>52)&2047)==2047)return NAN;
    ulong elapsed=atlas_clock_sub(exact_time,last);if(((elapsed>>52)&2047)==2047)return NAN;
    ulong rhs=m[slot+2]?m[slot+3]:atlas_clock_from_float(value[m[slot]]);
    z=float(atlas_clock_compare(elapsed,rhs,m[slot+1]));break;}
   case 16:z=float(state[a]==0);break;case 18:z=time;break;
   case 19:z=p[noise+a];break;case 23:z=p[clocks+a];break;
   case 3:z=value[a]+value[b];break;case 4:z=value[a]-value[b];break;
   case 5:z=value[a]*value[b];break;case 6:z=value[a]/value[b];break;
   case 7:z=-value[a];break;case 8:z=exp(value[a]);break;case 9:z=log(value[a]);break;
   case 10:z=tanh(value[a]);break;case 11:z=sqrt(value[a]);break;
   case 12:z=sin(value[a]);break;case 13:z=cos(value[a]);break;
   case 52:z=atlas_dynamic_math(slot,value[a]);break;
   case 53:{
    ulong cached=0;
    if(m[header]&1024){cached=atlas_dynamic_cache_slot(p,noise,b);
     if(cache[cached]==1){
      if(backward&&atlas_dynamic_cache_origin(p,cache,noise,b)&&atlas_dynamic_poisson_bits(value[a])!=atlas_dynamic_poisson_bits(cache[cached+2]))return NAN;
      z=cache[cached+1];break;
     }
    }
    ulong key=0;for(uint word=0;word<4;word++)key|=ulong(p[noise+16+4*b+word])<<(16*word);
    uint draws=0,error=0;int count=0;
    bool forced=false;
    if((m[header]&512)&&m[control+5]){
     if(m[header]&1024)forced=cached==atlas_dynamic_cache_slot(p,m[control+5]-1,m[control+6]);
     else forced=m[control+5]==noise+1&&m[control+6]==b;
    }
    if(forced){if((atlas_dynamic_poisson_bits(value[a])&0x7fffffffu)!=0)return NAN;count=1;}
    else count=atlas_dynamic_poisson(value[a],key,draws,error);
    if(error)return NAN;z=atlas_dynamic_bits(count);
    if((m[header]&1024)&&cache_store){cache[cached]=1;cache[cached+1]=z;cache[cached+2]=value[a];cache[cached+3]=atlas_dynamic_bits(int(uint(noise+1)));}
    break;
   }
   case 14:z=c==0?1:pow(abs(value[a]),c);if(value[a]<0){if(floor(c)!=c)return NAN;if(fmod(abs(c),2.0f)==1)z=-z;}break;
   case 20:z=value[a]<=value[b]?value[a]:value[b];break;
   case 21:z=value[a]>=value[b]?value[a]:value[b];break;
   case 22:z=value[value[a]==1?b:slot];break;
   case 54:case 55:z=value[b];break;
   case 56:z=value[a]*value[b];break;case 57:z=value[a]+value[b]-value[a]*value[b];break;
   case 26:z=float(value[a]>0);break;case 27:z=float(value[a]>=0);break;
   case 28:z=value[a]==0?0:value[b];break;case 29:z=value[a]==1?1:value[b];break;
   case 30:z=1-value[a];break;case 31:z=float(value[a]==value[b]);break;case 32:z=float(value[a]!=value[b]);break;
   case 33:z=atlas_dynamic_bits(int(uint(a)));break;case 34:z=state[a];break;
   case 35:z=weights[slot];break;case 36:z=weights[slot+parameter_index];break;case 37:z=weights[slot+m[a+parameter_index]];break;
   case 38:if(value[a]<-2147483648.0f||value[a]>=2147483648.0f)return NAN;z=atlas_dynamic_bits(int(value[a]));break;
   case 39:z=float(atlas_dynamic_int(value[a]));break;
   case 40:{int ia=atlas_dynamic_int(value[a]),ib=atlas_dynamic_int(value[b]);
    int r=0;
    if(slot>=7){
     if(slot==7)r=ia&ib;else if(slot==8)r=ia|ib;else if(slot==9)r=ia^ib;
     else if(slot==10||slot==11){
      if(ib<0||ib>=32)return NAN;
      if(slot==10)r=int(uint(ia)<<uint(ib));
      else if(ib==0)r=ia;
      else r=int((uint(ia)>>uint(ib))|(ia<0?(~0u<<(32u-uint(ib))):0u));
     }else return NAN;
    }else if(slot>=5){
     if(ib==0)return NAN;
     if(ia==(-2147483647-1)&&ib==-1)r=slot==5?ia:0;
     else{int q=ia/ib,rem=ia%ib;if(rem!=0&&((rem<0)!=(ib<0))){q-=1;rem+=ib;}r=slot==5?q:rem;}
    }else r=slot==0?int(uint(ia)+uint(ib)):slot==1?int(uint(ia)-uint(ib)):slot==2?int(uint(ia)*uint(ib)):slot==3?(ia<ib?ia:ib):(ia>ib?ia:ib);
    z=atlas_dynamic_bits(r);break;}
   case 41:{int ia=atlas_dynamic_int(value[a]),ib=atlas_dynamic_int(value[b]);
    z=float(slot==0?ia>ib:slot==1?ia>=ib:slot==2?ia<ib:slot==3?ia<=ib:slot==4?ia==ib:ia!=ib);break;}
   case 42:z=value[value[a]==1?b:slot];break;
   case 43:z=float(slot==0?value[a]>value[b]:slot==1?value[a]>=value[b]:slot==2?value[a]<value[b]:slot==3?value[a]<=value[b]:slot==4?value[a]==value[b]:value[a]!=value[b]);break;
   case 44:z=float(value[a]!=0);break;case 45:z=atlas_dynamic_bits(int(0u-uint(atlas_dynamic_int(value[a]))));break;
   case 46:if(value[b]==0)return NAN;z=floor(value[a]/value[b]);break;
   case 47:{if(value[b]==0)return NAN;float r=fmod(value[a],value[b]);z=r+((r!=0&&((r<0)!=(value[b]<0)))?value[b]:0.0f*value[b]);break;}
   default:return NAN;
  }
  bool integer=(op>=33&&op<=38)||op==40||op==42||op==45||op==49||op==50||op==53||op==55;
  if(!integer&&!isfinite(z))return NAN;value[j]=z;seen[j]=true;if(lazy)depth--;else cursor++;
 }
 if(!backward){valid=true;return value[count-1];}adjoint[count-1]=seed;
 if(score_enabled)for(ulong j=0;j<count;j++){
  ulong q=start+4*j,op=m[q],rate=m[q+1],stream=m[q+2];
  if(op!=53||!seen[j]||(scored&(1u<<uint(stream))))continue;
  if((m[header]&1024)&&!atlas_dynamic_cache_origin(p,cache,noise,stream))continue;
  scored|=1u<<uint(stream);
  if(m[q+3]){
   if((atlas_dynamic_poisson_bits(value[rate])&0x7fffffffu)==0){
    float unit[128];for(ulong k=0;k<count;k++)unit[k]=0;unit[rate]=1;
    bool needed=false;
    if(!atlas_dynamic_equation_vjp(m,p,start,constants,count,parameter_index,value,seen,unit,
      gradient,gradient_base,state_gradient,sources,context_count,true,needed,activity))return NAN;
    if(needed){
     if(probe_only)needed_mask|=1u<<uint(stream);
     else{
      if(!(m[header]&512))return NAN;
      float ready=p[noise+96];if(!isfinite(ready)||ready<0||ready>65535||ready!=floor(ready)||!(uint(ready)&(1u<<uint(stream))))return NAN;
      float alternate=p[noise+80+stream];if(!isfinite(alternate))return NAN;
      adjoint[rate]+=alternate/float(m[0])-score_loss;
     }
    }
   }else if(!probe_only){
    float score=atlas_dynamic_poisson_score(value[rate],atlas_dynamic_int(value[j]));
    if(!isfinite(score))return NAN;adjoint[rate]+=score_loss*score;
   }
  }
 }
 if(score_enabled&&probe_only){valid=true;return value[count-1];}
 bool ignored=false;
 if(!atlas_dynamic_equation_vjp(m,p,start,constants,count,parameter_index,value,seen,adjoint,
  gradient,gradient_base,state_gradient,sources,context_count,false,ignored,activity))return NAN;
 valid=true;return value[count-1];
}

float atlas_dynamic_equation(device const ulong* m,device const float* p,device const float* weights,
 ulong header,ulong parameter_index,float time,ulong exact_time,ulong clocks,ulong noise,
 thread const float* state,device float* gradient,ulong gradient_base,
 float seed,bool backward,thread float* state_gradient,thread bool& valid){
 // Protected legacy v4 mode uses the canonical mask suffix and local states.
 // Real v5 reverse callers supply their own activity mode and taped addresses.
 int sources[1]={0};uint scored=0,needed_mask=0;return atlas_dynamic_equation_scored(m,p,weights,header,parameter_index,time,exact_time,clocks,noise,
  state,gradient,gradient_base,seed,backward,state_gradient,valid,0,false,scored,needed_mask,sources,0,(m[9]&2)?2u:0u,gradient,false);
}

// v5r5 optional addressing metadata, stored in reserved action word 15.
// Table indices are exact int32 payloads; physical addresses fit in float32.
int atlas_dynamic_address(device const ulong* m,device const float* live,ulong base,
 int index,ulong depth,ulong tables){
 int target=-1;
 for(ulong d=0;d<depth;d++){
  ulong length=m[tables+2*d],data=m[tables+2*d+1];
  if(index<0||ulong(index)>=length)return -1;
  target=int(m[data+ulong(index)]);
  if(d+1<depth)index=atlas_dynamic_int(live[base+ulong(target)]);
 }
 return target;
}
void atlas_dynamic_gather(device const ulong* m,ulong h,device const float* live,ulong base,
 device float* tape,ulong at,thread float* context){
 ulong c=m[h+1],r=m[h+2],desc=m[h+15];
 for(ulong j=0;j<c;j++){
  int target=int(m[r+j]);
  if(desc&&m[h+10]){
   ulong entry=m[m[desc]+j];
   if(entry)target=atlas_dynamic_address(m,live,base,atlas_dynamic_int(live[base+m[entry]]),m[entry+1],entry+2);
  }
  context[j]=target>=0?live[base+ulong(target)]:0;
  tape[at+j]=context[j];if(desc)tape[at+c+j]=float(target);
 }
}
bool atlas_dynamic_reads_valid(device const ulong* m,ulong h,device const float* tape,ulong at){
 if(!m[h+15])return true;ulong c=m[h+1];
 for(ulong j=0;j<c;j++)if(tape[at+c+j]<0)return false;
 return true;
}
bool atlas_dynamic_index_output(device const ulong* m,ulong h,ulong output){
 ulong desc=m[h+15];if(!desc)return false;
 for(ulong j=0;j<m[h+3];j++){ulong entry=m[m[desc+1]+j];if(entry&&m[entry]==1&&m[entry+1]==output)return true;}
 return false;
}
bool atlas_dynamic_outputs(device const ulong* m,device const float* p,device const float* weights,
 ulong h,float gate,float time,ulong exact_time,ulong clock,ulong noise,thread const float* context,
 device float* gradient,ulong gb,thread float* out,thread float* gstate,device float* cache){
 for(ulong j=0;j<m[h+3];j++){
  out[j]=0;if(gate==0&&!atlas_dynamic_index_output(m,h,j))continue;
  bool valid=false;
  int sources[1]={0};uint scored=0,needed_mask=0;
  float value=atlas_dynamic_equation_scored(m,p,weights,m[h+5]+3*j,m[h+11],time,exact_time,clock,noise,context,gradient,gb,0,false,gstate,valid,0,false,scored,needed_mask,sources,0,0,cache,gate!=0);
  if(gate==0){out[j]=valid?value:atlas_dynamic_bits(-2147483647-1);continue;}
  ulong target=m[m[h+4]+j];
  if(!valid||((!m[28]||!m[m[28]+target])&&!isfinite(value))||(m[m[17]+target]&&value!=0&&value!=1))return false;
  out[j]=value;
 }
 return true;
}
bool atlas_dynamic_targets(device const ulong* m,ulong h,device const float* live,ulong base,
 device float* tape,ulong at,thread const float* context,thread const float* out){
 ulong desc=m[h+15];if(!desc)return true;
 ulong c=m[h+1],w=m[h+3];bool valid=true;
 for(ulong j=0;j<w;j++){
  int target=int(m[m[h+4]+j]);ulong entry=m[m[desc+1]+j];
  if(entry){
   int index=atlas_dynamic_int(m[entry]==0?context[m[entry+1]]:out[m[entry+1]]);
   target=atlas_dynamic_address(m,live,base,index,m[entry+2],entry+3);
  }
  tape[at+2*c+j]=float(target);tape[at+2*c+w+j]=target>=0?live[base+ulong(target)]:0;
  valid=valid&&target>=0;
 }
 return valid;
}
int atlas_dynamic_target(device const ulong* m,ulong h,device const float* tape,ulong at,ulong j){
 return m[h+15]?int(tape[at+2*m[h+1]+j]):int(m[m[h+4]+j]);
}
bool atlas_dynamic_winner(device const ulong* m,ulong h,device const float* tape,ulong at,ulong j){
 if(!m[h+15])return true;int target=atlas_dynamic_target(m,h,tape,at,j);if(target<0)return false;
 for(ulong k=j+1;k<m[h+3];k++)if(atlas_dynamic_target(m,h,tape,at,k)==target)return false;
 return true;
}
bool atlas_dynamic_reverse(device const ulong* m,device const float* p,device const float* weights,
 ulong h,device float* tape,ulong at,device const float* adjoint,ulong base,
 float gate,bool differentiate,float score_loss,float time,ulong exact_time,ulong clock,ulong noise,thread const float* context,
 device float* gradient,ulong gb,thread float* gstate,thread float* old_gradient,thread float& dg){
 ulong c=m[h+1],w=m[h+3];dg=0;
 for(ulong j=0;j<c;j++)gstate[j]=0;
 for(ulong j=0;j<w;j++)old_gradient[j]=0;
 if(m[h+15]&&gate==0&&!differentiate)return true;
 if(m[h+15]){
  bool invalid=false;for(ulong j=0;j<w;j++)invalid=invalid||atlas_dynamic_target(m,h,tape,at,j)<0;
  if(invalid){for(ulong k=0;k<m[4];k++)if(adjoint[base+k]!=0)return false;return true;}
 }
 int sources[64];for(ulong j=0;j<c;j++)sources[j]=m[h+15]?int(tape[at+c+j]):int(m[m[h+2]+j]);
 if(gate!=0){
  uint scored=0,needed_mask=0;
  for(ulong j=0;j<w;j++){
   ulong header=m[h+5]+3*j;if(!(m[header]&256))continue;
   if(!atlas_dynamic_reads_valid(m,h,tape,at))return false;
   bool valid=false;atlas_dynamic_equation_scored(m,p,weights,header,m[h+11],time,exact_time,clock,noise,
    context,gradient,gb,0,true,gstate,valid,score_loss,true,scored,needed_mask,sources,c,true,tape,false);
   if(!valid)return false;
  }
  if(m[9]==2&&(m[m[h+5]]&512))tape[at+c+(m[h+15]?c+2*w:0)]=float(needed_mask);
 }
 if(m[9]==2)return true;
 for(ulong j=0;j<w;j++){
  if(!atlas_dynamic_winner(m,h,tape,at,j))continue;
  ulong target=ulong(atlas_dynamic_target(m,h,tape,at,j));if(m[m[16]+target])continue;
  float g=adjoint[base+target];if(g==0)continue;
  if(!atlas_dynamic_reads_valid(m,h,tape,at))return false;
  old_gradient[j]=g*(1-gate);
  if(gate!=0){
   uint scored=0,needed_mask=0;bool valid=false;
   atlas_dynamic_equation_scored(m,p,weights,m[h+5]+3*j,m[h+11],time,exact_time,clock,noise,
    context,gradient,gb,g*gate,true,gstate,valid,0,false,scored,needed_mask,sources,c,true,tape,false);
   if(!valid)return false;
  }
  if(differentiate){
   bool valid=false;uint scored=0,needed_mask=0;
   float value=atlas_dynamic_equation_scored(m,p,weights,m[h+5]+3*j,m[h+11],time,exact_time,clock,noise,context,gradient,gb,0,false,gstate,valid,0,false,scored,needed_mask,sources,c,0,tape,false);
   if(!valid)return false;
   float old=0;
   if(m[h+15])old=tape[at+2*c+w+j];
   else{ulong k=0;while(m[m[h+2]+k]!=target)k++;old=context[k];}
   dg+=g*(value-old);
  }
 }
 return true;
}
void atlas_dynamic_apply_reverse(device const ulong* m,ulong h,device const float* tape,ulong at,
 device float* adjoint,ulong base,thread const float* gstate,thread const float* old_gradient){
 ulong c=m[h+1],w=m[h+3];
 for(ulong j=0;j<w;j++)if(atlas_dynamic_winner(m,h,tape,at,j)){
  ulong target=ulong(atlas_dynamic_target(m,h,tape,at,j));if(!m[m[16]+target])adjoint[base+target]=old_gradient[j];
 }
 for(ulong j=0;j<c;j++){
  int source=m[h+15]?int(tape[at+c+j]):int(m[m[h+2]+j]);
  if(source>=0&&!m[m[16]+ulong(source)])adjoint[base+ulong(source)]+=gstate[j];
 }
}

// v5r8: m[30] stores main frame count then visit frame+1 and active clocks;
// m[31] maps actions to clocks. Idle visits have no loss seed or TBPTT cut.
ulong atlas_dynamic_frame(device const ulong* m,ulong t){return m[m[30]+1+t*(m[23]+1)];}
bool atlas_dynamic_active(device const ulong* m,ulong t,ulong a){return m[m[30]+2+t*(m[23]+1)+m[m[31]+a]]!=0;}
bool atlas_dynamic_cut(device const ulong* m,ulong t){ulong f=atlas_dynamic_frame(m,t);return m[10]&&f>1&&(f-1)%m[10]==0;}

kernel void atlas_dynamic_bptt(
 device const ulong* m [[buffer(0)]],device const float* p [[buffer(1)]],
 device const float* input [[buffer(2)]],device const float* weights [[buffer(3)]],
 device const float* initial [[buffer(4)]],device const ulong* labels [[buffer(5)]],
 device float* tape [[buffer(6)]],device float* spikes [[buffer(7)]],
 device float* live [[buffer(8)]],device float* gradient [[buffer(9)]],
 device float* adjoint [[buffer(10)]],device float* logits [[buffer(11)]],
 device float* losses [[buffer(12)]],device float* ds [[buffer(13)]], uint b [[thread_position_in_grid]]) {
 ulong B=m[0],T=m[1],I=m[2],N=m[3],W=m[4],E=m[5],C=m[6],A=m[7],Q=m[8];if(b>=B)return;
 ulong base=ulong(b)*W,gb=ulong(b)*E,sb=ulong(b)*N;losses[b]=0;
 for(ulong k=0;k<W;k++){live[base+k]=initial[base+k];adjoint[base+k]=0;}
 for(ulong k=0;k<E;k++)gradient[gb+k]=0;
 for(ulong k=0;k<T*N;k++)spikes[ulong(b)*T*N+k]=0;
 for(ulong k=0;k<T*Q;k++)tape[ulong(b)*T*Q+k]=0;
 atlas_dynamic_cache_initialize(m,p,tape,b);
 for(ulong j=0;j<C;j++)logits[ulong(b)*C+j]=0;
 for(ulong t=0;t<T;t++){
  ulong nt=(ulong(b)*T+t)*N,clock=m[22]+t*m[23];
  for(ulong a=0;a<A;a++){
   if(!atlas_dynamic_active(m,t,a))continue;
   ulong cid=m[m[31]+a];float time=p[clock+cid];ulong exact_time=m[29]?m[m[29]+t*m[23]+cid]:0;
   ulong h=m[13]+16*a,c=m[h+1],r=m[h+2],w=m[h+3],wr=m[h+4],at=(ulong(b)*T+t)*Q+m[h];
   float context[64],out[64],gstate[64];
   atlas_dynamic_gather(m,h,live,base,tape,at,context);for(ulong j=0;j<c;j++)gstate[j]=0;
   if(m[h+6]){
    ulong k=m[h+6]-1,ref=m[h+14]?0:m[m[18]+k];float theta=m[h+14]?0:ref?weights[ref-1]:p[m[19]+k];
    if((c>1&&context[1]!=0&&context[1]!=1)||(m[h+14]==3&&context[0]!=0&&context[0]!=1)){losses[b]=NAN;return;}
    spikes[nt+k]=float((c==1||context[1]==1)&&(m[h+14]==2?context[0]>=theta:context[0]>theta));if(m[32])live[base+m[m[32]+k]]=spikes[nt+k];continue;
   }
   if(!m[h+10])continue;
   ulong kind=m[h+7],index=m[h+8];float gate=kind==0?1:kind==1?input[(ulong(b)*T+t)*I+index]:kind==2?spikes[nt+index]:context[index];
   if(gate!=0&&!atlas_dynamic_reads_valid(m,h,tape,at)){losses[b]=NAN;return;}
   ulong noise=m[24]+(ulong(b)*T+t)*m[25]+m[h+12];
   if(!atlas_dynamic_outputs(m,p,weights,h,gate,time,exact_time,clock,noise,context,gradient,gb,out,gstate,tape)){losses[b]=NAN;return;}
   bool targets=atlas_dynamic_targets(m,h,live,base,tape,at,context,out);
   if(gate!=0){
    if(!targets){losses[b]=NAN;return;}
    for(ulong j=0;j<w;j++)live[base+ulong(atlas_dynamic_target(m,h,tape,at,j))]=out[j];
   }
  }
  for(ulong j=0;j<C;j++)logits[ulong(b)*C+j]+=spikes[nt+N-C+j]*p[2]/float(m[m[30]]);
 }
 float maximum=-INFINITY,total=0;
 for(ulong j=0;j<C;j++)maximum=max(maximum,logits[ulong(b)*C+j]);
 for(ulong j=0;j<C;j++)total+=exp(logits[ulong(b)*C+j]-maximum);
 losses[b]=maximum+log(total)-logits[ulong(b)*C+labels[b]];if(!m[9])return;
 for(ulong t=T;t-->0;){
  ulong nt=(ulong(b)*T+t)*N,clock=m[22]+t*m[23];
  for(ulong k=0;k<N;k++)ds[sb+k]=0;
  if(m[32]||atlas_dynamic_frame(m,t))for(ulong j=0;j<C;j++)ds[sb+N-C+j]=(exp(logits[ulong(b)*C+j]-maximum)/total-float(j==labels[b]))/float(B)*p[2]/float(m[m[30]]);
  for(ulong a=A;a-->0;){
   if(!atlas_dynamic_active(m,t,a))continue;
   ulong cid=m[m[31]+a];float time=p[clock+cid];ulong exact_time=m[29]?m[m[29]+t*m[23]+cid]:0;
   ulong h=m[13]+16*a,c=m[h+1],r=m[h+2],w=m[h+3],wr=m[h+4],at=(ulong(b)*T+t)*Q+m[h];
   float context[64],gstate[64],old_gradient[64];
   for(ulong j=0;j<c;j++){context[j]=tape[at+j];gstate[j]=0;}
   if(m[h+6]){
    if(m[9]==2)continue;
    ulong k=m[h+6]-1,ref=m[h+14]?0:m[m[18]+k];float theta=m[h+14]?0:ref?weights[ref-1]:p[m[19]+k];
    float seed=ds[sb+k];if(m[32]){ulong cell=base+m[m[32]+k];seed+=adjoint[cell];adjoint[cell]=0;}
    if(c==1||context[1]==1){
     float z=1+p[0]*abs(context[0]-theta),g=m[h+14]==3?seed:seed*p[1]/(z*z);
     adjoint[base+m[r]]+=g;if(ref)gradient[gb+ref-1]-=g;
    }
    continue;
   }
   if(!m[h+10])continue;
   ulong kind=m[h+7],index=m[h+8];float gate=kind==0?1:kind==1?input[(ulong(b)*T+t)*I+index]:kind==2?spikes[nt+index]:context[index];
   bool gate_gradient=!m[h+9]&&(kind==2||(kind==3&&!m[m[16]+m[r+index]]));float dg=0;
   ulong noise=m[24]+(ulong(b)*T+t)*m[25]+m[h+12];
   if(m[h+15]&&gate==0&&!gate_gradient)continue;
   if(!atlas_dynamic_reverse(m,p,weights,h,tape,at,adjoint,base,gate,gate_gradient,losses[b]/float(B),time,exact_time,clock,noise,context,gradient,gb,gstate,old_gradient,dg)){losses[b]=NAN;return;}
   if(m[9]==2)continue;
   atlas_dynamic_apply_reverse(m,h,tape,at,adjoint,base,gstate,old_gradient);
   if(kind==2)ds[sb+index]+=dg;else if(kind==3&&!m[m[16]+m[r+index]])adjoint[base+m[r+index]]+=dg;
  }
  if(atlas_dynamic_cut(m,t))for(ulong k=0;k<W;k++)adjoint[base+k]=0;
 }
 if(m[9]!=2&&m[26])for(ulong k=0;k<W;k++){ulong ref=m[m[27]+k];if(ref)gradient[gb+ref-1]+=adjoint[base+k];}
}

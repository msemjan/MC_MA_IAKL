#ifndef CUDA_UTILITIES_H_
#define CUDA_UTILITIES_H_

///
/// Makes a snapshot of the lattice, saves the configurations of up and down 
/// spins to files "up.txt" and "down.txt", respectively. And at the end it 
/// plots snapshot via pipe to the gnuplot.
///
void makeSnapshot( Lattice* s, FILE* gnuplotPipe, std::string path ){
    std::ofstream up(path+"up.txt", std::ios::out);
    std::ofstream down(path+"down.txt", std::ios::out);
    
    int x, y;

    for(int i = 0; i < L; i++){
        for(int j = 0; j < L; j++){
            // s1
            x = 4*(j+1) - 2*(i+1);
            y = -2*(i+1);
            if(d_s->s1[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
            // s2
            x = 4*(j+1) + 2 - 2*(i+1);
            y = -2*(i+1);
            if(d_s->s2[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
            // s3
            x = 4*(j+1)-(2*(i+1)+1);
            y = -2*(i+1)-1;
            if(d_s->s3[i*L+j]==-1){
               down << x << " " << y << std::endl;
            }else{
               up << x << " " << y << std::endl;
            }
        }
    }
    up.close();
    down.close();

    fprintf(gnuplotPipe,"%s", ("plot \"" + (path+"up.txt") + "\" with circles"+
           " linecolor rgb \"#ff0000\" fill solid,\\\n" +
           "\"" + (path+"down.txt") + "\" with circles linecolor rgb " + 
           "\"#0000ff\"" + " fill solid\n").c_str());
    fprintf(gnuplotPipe, "%s", ("set xrange [-"+std::to_string(2*L)+":"+
                std::to_string(4.5*L)+"]\n").c_str());
    fprintf(gnuplotPipe, "%s", ("set yrange [-"+std::to_string(3*L)+":"+
                std::to_string(0)+"]\n").c_str());
    fflush(gnuplotPipe);
}
    
#endif

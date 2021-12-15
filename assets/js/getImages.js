$(document).ready(function() {

    function addImage(imageUrl, title)
    {
        var img = `<div class="media">`+
            `<div class="overlay"></div>`+
            `<img src=${imageUrl} alt="">`+
            `<div class="image-details">`+
            `<p>${title}</p>`+
            `</div>`+
        `</div>`;
        return img;
    }

    function getImages(model_name, optim_name, n_in)
    {
        var sgd_lrs = [0.2, 0.3, 0.4, 0.01, 0.001, 0.0001];
        var lrs = [0.1, 0.01, 0.001, 0.0001];

        var lrs = {
            'SGD': sgd_lrs,
            'LBFGS': lrs,
            'Adam': lrs,
            'NonlinearCG': lrs
        }
        var predict_imgs = [];
        var loss_imgs = [];
        lrs[optim_name].forEach(lr=>{
            var file_name = `${model_name}_${optim_name}_${lr}_${n_in}`;
            
            if(optim_name === "NonlinearCG"){
                var maxits = [1,3000];
                var beta_types = ["HS", "FR_PR"];
                maxits.forEach(maxit=>{
                    beta_types.forEach(beta_type=>{
                        var file_name_org = file_name + `_${maxit}_${beta_type}`;
                        var loss_fig_path = `../../logs/loss_log/${optim_name}/` + file_name_org;
                        var predict_fig_path = `../../logs/figures/${optim_name}/` + file_name_org;

                        // var loss_fig_path = `/logs/loss_log/${optim_name}/` + file_name_org;
                        // var predict_fig_path = `/logs/figures/${optim_name}/` + file_name_org;
                        predict_imgs.push(predict_fig_path);
                        loss_imgs.push(loss_fig_path);
                        
                    });
                });
            }
            else{
                var loss_fig_path = `../../logs/loss_log/${optim_name}/` + file_name;
                var predict_fig_path = `../../logs/figures/${optim_name}/` + file_name;
                // var loss_fig_path = `/logs/loss_log/${optim_name}/` + file_name;
                // var predict_fig_path = `/logs/figures/${optim_name}/` + file_name;
                predict_imgs.push(predict_fig_path);
                loss_imgs.push(loss_fig_path);
            }
            
            
        });

        var predict_name = `${model_name}${n_in}-${optim_name}-Predict`;
        $(`#${predict_name}`).append(`<h1>${predict_name}</h1>`);
        predict_imgs.forEach(predict_img_url=>{
            var description = predict_img_url.split('/').pop();
            $(`#${predict_name}`).append(addImage(predict_img_url + ".png", description));
        });
        var loss_name = `${model_name}${n_in}-${optim_name}-Loss`;
        $(`#${loss_name}`).append(`<h1>${loss_name}</h1>`);
        loss_imgs.forEach(loss_img_url=>{
            var description = loss_img_url.split('/').pop();
            $(`#${loss_name}`).append(addImage(loss_img_url + ".png", description));
        });
}

    function updateImages(){
        var models = ["MLP", "MLP_Large", "MLP_Multistep"];
        var optims = ["LBFGS", "NonlinearCG", "SGD", "Adam"];
        models.forEach(model=>{
            optims.forEach(optim=>{
                if(model === "MLP_Multistep"){
                    var n_ins = [3,5,10,20];
                    n_ins.forEach(n_in=>{
                        getImages(model, optim, n_in);
                    });
                }
                else{
                    var n_in = 2;
                    getImages(model, optim, n_in);
                }
            })
        })
    }

    updateImages();

})
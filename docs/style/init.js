$(document).ready(function(){
    $(".button-collapse").sideNav();
    $('.materialboxed').materialbox();
    $('.scrollspy').scrollSpy();
    $('select').material_select();


}); // end of document ready

function change_scale_img() {

    var img = document.getElementsByName("demo-img")[0];
    var method = img.id.split("-")[1];
    change_method(method);
}

function change_method(method) {

    var x = document.getElementById("scale");
    var i = x.selectedIndex;
    var scale = x.options[i].value;

    x = document.getElementById("img-name");
    i = x.selectedIndex;
    var name = x.options[i].value;

    
    var img = document.getElementsByName("demo-img")[0];
    var filename = name + "_" + scale + "_" + method;
    img.src = scale + "/" + filename + ".png";
    img.id = "img-" + method;

    var caption = document.getElementById("img-caption");
    caption.innerHTML = filename;
    
}
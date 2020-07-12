
/* 카드 변경 함수 모듈화
function change_card(id, index) {
    $('#det-item-img-0').attr("src", pass_img_list[index]);
    $('#det-review-id-0').text(pass_id_list[index]);
    $('#det-review-txt-0').text(pass_comment_list[index]);
}*/


$(document).ready(function () {
    var upload_img
    var state = $('.info-btn').attr('value');
    var uploaded_img = $("#upload-img").attr('src');
    var pass_id_list = ["juhoK", "hyemzzzy", "tyk13"];
    var fail_id_list = ["forizy", "mubigs20", "yZL"];
    var pass_comment_list = ["이정도면 만족하네요 좋아요", "옷이 너무 이뻐요 ^-^", "사이즈 적당한 것 같아요 괜찮네요~"];
    var fail_comment_list = ["흠 사이즈가 조금 큰 것 같네요..", "나들이 나가야겠어요~", "색깔이 맘에 들어요"];
    var pass_img_list = ['/static/img/pass_ex1.jpg', '/static/img/pass_ex2.jpg', '/static/img/pass_ex3.jpg'];
    var fail_img_list = ['/static/img/fail_ex1.jpg', '/static/img/fail_ex2.jpg', '/static/img/fail_ex3.jpg'];
    if (state == "PASS :)") {
        pass_img_list[1] = uploaded_img;
    }
    else {
        fail_img_list[1] = uploaded_img;
    }


    //nav-item 클릭시 활성화 이벤트
    $('#mubigs_navbar .nav-item').on('click', function (event) {
        //active 클래스 부여
        if (!$(this).hasClass('active')) {
            $('#mubigs_navbar .nav-item').removeClass('active');
            $(this).addClass('active');
        }

        //scroll 이동
        if ($(this).attr('id') == "nav-item-1") {
            $('html').animate({scrollTop:$('body').offset().top}, 400);
        }
        else if ($(this).attr('id') == "nav-item-2") {
            $('html').animate({scrollTop:$('#view-2').offset().top}, 400);
        }

    });

    //image upload container hover event

    //image 업로드 완료 시
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $('#upload-img').attr('src', e.target.result);
                upload_img = e.target.result;
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#file-input").change(function(){
        readURL(this);

        var img=$('#upload-img');

        img.on('load',function() {
            img.width('auto');
            img.height('300px');
        });

    });

    //submit 클릭시 서버로 이미지 보내기


    //prediction이 완료되었을 때 동작



    //nav-det-menu 클릭시 활성화 이벤트


    $('.nav-det-menu').on('click', function (event) {
        //active 클래스 부여
        if (!$(this).hasClass('active')) {
            $('.nav-det-menu').removeClass('active');
            $(this).addClass('active');

            //pass-fail 데이터 바꾸기
            if ($(this).attr('id') == "nav-det-menu-pass") {
                $('#det-item-img-0').attr("src", pass_img_list[0]);
                $('#det-item-img-1').attr("src", pass_img_list[1]);
                $('#det-item-img-2').attr("src", pass_img_list[2]);

                $('#det-review-id-0').text(pass_id_list[0]);
                $('#det-review-id-1').text(pass_id_list[1]);
                $('#det-review-id-2').text(pass_id_list[2]);

                $('#det-review-txt-0').text(pass_comment_list[0]);
                $('#det-review-txt-1').text(pass_comment_list[1]);
                $('#det-review-txt-2').text(pass_comment_list[2]);
            }
            else {
                $('#det-item-img-0').attr("src", fail_img_list[0]);
                $('#det-item-img-1').attr("src", fail_img_list[1]);
                $('#det-item-img-2').attr("src", fail_img_list[2]);

                $('#det-review-id-0').text(fail_id_list[0]);
                $('#det-review-id-1').text(fail_id_list[1]);
                $('#det-review-id-2').text(fail_id_list[2]);

                $('#det-review-txt-0').text(fail_comment_list[0]);
                $('#det-review-txt-1').text(fail_comment_list[1]);
                $('#det-review-txt-2').text(fail_comment_list[2]);
            }
        }

    });


    //det-card pre, nxt 버튼 클릭 이벤트

});



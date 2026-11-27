# GitHub 교육용 HTML 작성 규칙

이 저장소에서 새 HTML 교재를 만들거나 기존 교재를 수정할 때는 먼저 다음 문서를 확인한다.
- https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/STANDARD.md
- https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/INTEGRATION.md

적용 사항:
1. 책장 배너 이미지는 중앙 표준의 hero-bookshelf.jpg와 동일하게 유지한다.
2. 좌측 전체 장 목차와 현재 페이지 절 목차 및 모바일 메뉴를 구현한다.
3. 기존 학습 내용·코드·이미지는 보존하고, 차이가 있는 부분만 수정한다.
4. 수정 시 장별 파일명을 _v01 → _v02로 올리고 이전 생성본은 결과 폴더에서 정리한다.
5. 코드 문법, 내부 링크, 개인정보, 접근성과 브라우저 표시를 검증한다.
6. 확인하지 못한 실행 결과는 미검증으로 표기한다.
7. 원격 문서를 읽을 수 없으면 추정 적용하지 않고 로컬 규칙 사본을 요구한다.

이 저장소 추가 규칙:
1. 원고는 `content_src/`에서 고친다. `content/`는 `tools/expand_code.py` 생성물이다.
2. 장 순서와 장 안의 순서는 원본 강의·실습 자료 순서를 따른다. 개념별로 재배치하지 않는다.
3. 본문 코드는 손으로 옮겨 적지 않는다. `code/*.py`의 `# %% CODE NN` 블록을 `⟪code:파일:NN⟫`으로 불러오고, 실행 결과는 `tools/run_blocks.py`가 만든 `code/out/*.json`을 쓴다.
4. 원본 실습 밖의 확인 코드는 `code/chNN_extra_checks.py`에 두고 본문에 "추가 확인"으로 밝힌다.
5. 그림은 `tools/make_figures_chNN.py`가 만드는 SVG가 원본이다. 그림을 고치면 스크립트를 고친다.
6. 원본 자료와 달라진 점은 장 끝 "원본 자료와 달라진 점" 표에 적는다.
